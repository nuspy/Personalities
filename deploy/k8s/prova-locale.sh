#!/usr/bin/env bash
# La prova in locale: la piattaforma intera su un k3s dentro Docker, con Istio,
# le immagini costruite sulla macchina e l'overlay `locale`.
#
#   deploy/k8s/prova-locale.sh tutto      # avvia, carica, installa, applica, verifica
#   deploy/k8s/prova-locale.sh verifica   # solo le verifiche, su un cluster già su
#   deploy/k8s/prova-locale.sh ferma      # toglie il cluster di prova
#
# **Non tocca nessun altro cluster.** Ogni comando passa `--kubeconfig` con il
# file di questo cluster, e prima di applicare qualcosa lo script controlla che
# punti a 127.0.0.1: un contesto predefinito dimenticato verso la produzione
# non viene mai usato.
#
# Nessun modello viene interrogato: le verifiche guardano che i servizi
# partano, si parlino in mTLS e rispondano sui loro domini — non la qualità
# delle risposte, che si prova con i test.
set -euo pipefail

# Git Bash su Windows traduce ogni argomento che somiglia a un percorso — anche
# `/etc/...` dentro un container — in un percorso di Windows. Si spegne, e si
# convertono esplicitamente solo i percorsi della macchina montati come volumi.
export MSYS_NO_PATHCONV=1
percorso_host() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }

QUI="$(cd "$(dirname "$0")" && pwd)"
NOME=persona-k3s
K3S_IMMAGINE=rancher/k3s:v1.31.4-k3s1
ISTIO_VERSIONE=1.24.2
KUBECONFIG_LOCALE="$QUI/.kube-locale.yaml"
# Lo stesso cluster visto da dentro la sua rete, per istioctl.
KUBECONFIG_INTERNO="$QUI/.kube-locale-interno.yaml"
# Le porte sulla macchina: il gateway (i domini di overlays/locale la citano)
# e l'API del cluster, spostata dalla 6443 che altri Kubernetes locali usano.
# Mai la 8000: è di altri progetti.
PORTA=8088
PORTA_API=16443
IMMAGINI=(persona-api:locale persona-worker:locale persona-web:locale persona-admin:locale)

k() { kubectl --kubeconfig "$(percorso_host "$KUBECONFIG_LOCALE")" "$@"; }

controlla_cluster() {
  local server
  server="$(k config view --minify -o jsonpath='{.clusters[0].cluster.server}')"
  if [[ "$server" != "https://127.0.0.1:$PORTA_API" ]]; then
    echo "Il kubeconfig punta a $server e non al cluster locale: mi fermo." >&2
    exit 1
  fi
}

avvia() {
  if docker ps --format '{{.Names}}' | grep -qx "$NOME"; then
    echo "Il cluster $NOME è già acceso."
  else
    for porta in "$PORTA" "$PORTA_API"; do
      if (exec 3<>"/dev/tcp/127.0.0.1/$porta") 2>/dev/null; then
        echo "La porta $porta è già in uso da qualcos'altro: mi fermo invece di contendergliela." >&2
        exit 1
      fi
    done
    # Traefik spento: l'ingresso è il gateway di Istio, che il bilanciatore
    # di k3s espone sulla porta 80 del nodo, cioè la $PORTA della macchina.
    docker run -d --name "$NOME" --privileged \
      -p "$PORTA_API:6443" -p "$PORTA:80" \
      "$K3S_IMMAGINE" server --disable=traefik --tls-san 127.0.0.1 >/dev/null
  fi
  echo "Attendo il kubeconfig…"
  for _ in $(seq 1 60); do
    if docker exec "$NOME" test -s /etc/rancher/k3s/k3s.yaml 2>/dev/null; then break; fi
    sleep 2
  done
  docker exec "$NOME" cat /etc/rancher/k3s/k3s.yaml > "$KUBECONFIG_INTERNO"
  sed "s#https://127.0.0.1:6443#https://127.0.0.1:$PORTA_API#" "$KUBECONFIG_INTERNO" > "$KUBECONFIG_LOCALE"
  controlla_cluster
  for _ in $(seq 1 60); do
    if k get nodes 2>/dev/null | grep -q " Ready"; then break; fi
    sleep 2
  done
  k get nodes
}

carica_immagini() {
  # Le immagini costruite sulla macchina, dentro il containerd del k3s: senza
  # registro, e con imagePullPolicy IfNotPresent nessuno prova a scaricarle.
  for immagine in "${IMMAGINI[@]}"; do
    docker image inspect "$immagine" >/dev/null 2>&1 || {
      echo "Manca l'immagine $immagine: costruiscila prima (vedi il README)." >&2
      exit 1
    }
  done
  docker save "${IMMAGINI[@]}" | docker exec -i "$NOME" ctr images import -
}

installa_istio() {
  controlla_cluster
  # istioctl dall'immagine ufficiale, nella rete del k3s: lì 127.0.0.1:6443 è
  # l'API del cluster, e sulla macchina non serve installare niente.
  docker run --rm --network "container:$NOME" \
    -v "$(percorso_host "$KUBECONFIG_INTERNO"):/kube/config:ro" \
    -v "$(percorso_host "$QUI/istio/istio-installazione.yaml"):/istio.yaml:ro" \
    -e KUBECONFIG=/kube/config \
    "istio/istioctl:$ISTIO_VERSIONE" install -f /istio.yaml -y
  k -n istio-system rollout status deploy/istiod --timeout=300s
  k -n istio-system rollout status deploy/istio-ingressgateway --timeout=300s
}

applica() {
  controlla_cluster
  kubectl kustomize --load-restrictor LoadRestrictionsNone "$(percorso_host "$QUI/overlays/locale")" | k apply -f -
  for risorsa in statefulset/postgres statefulset/redis deploy/keycloak deploy/api deploy/web deploy/admin deploy/worker-cpu deploy/otel-collector; do
    k -n personalities rollout status "$risorsa" --timeout=600s
  done
}

verifica() {
  controlla_cluster
  local falliti=0
  prova() {
    local descrizione="$1" atteso="$2" ottenuto="$3"
    if [[ "$ottenuto" == *"$atteso"* ]]; then
      echo "  ok   $descrizione"
    else
      echo "  NO   $descrizione — atteso «$atteso», ottenuto «$ottenuto»"
      falliti=$((falliti + 1))
    fi
  }
  url() { curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$1" || echo "errore"; }

  echo "Ingresso e instradamento"
  prova "API /healthz" '"status":"ok"' "$(curl -s --max-time 15 "http://api.personalities.localtest.me:$PORTA/healthz")"
  prova "API catalogo" "[" "$(curl -s --max-time 15 "http://api.personalities.localtest.me:$PORTA/personalities")"
  prova "sito" "200" "$(url "http://personalities.localtest.me:$PORTA/")"
  prova "console" "200" "$(url "http://admin.personalities.localtest.me:$PORTA/")"
  prova "Keycloak, emittente pubblico" "\"issuer\":\"http://auth.personalities.localtest.me:$PORTA/realms/personalities\"" \
    "$(curl -s --max-time 15 "http://auth.personalities.localtest.me:$PORTA/realms/personalities/.well-known/openid-configuration")"
  prova "Keycloak, console non esposta" "404" "$(url "http://auth.personalities.localtest.me:$PORTA/admin/")"
  prova "dominio sconosciuto rifiutato" "404" "$(url "http://sconosciuto.localtest.me:$PORTA/")"
  local accesso="http://auth.personalities.localtest.me:$PORTA/realms/personalities/protocol/openid-connect/auth?client_id=persona-frontend&response_type=code&scope=openid&code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM&code_challenge_method=S256"
  prova "login: ritorno verso il sito accettato" "kc-form-login" \
    "$(curl -s --max-time 15 "$accesso&redirect_uri=http%3A%2F%2Fpersonalities.localtest.me%3A$PORTA%2F" | grep -o kc-form-login | head -1)"
  prova "login: ritorno verso un dominio estraneo rifiutato" "Invalid parameter: redirect_uri" \
    "$(curl -s --max-time 15 "$accesso&redirect_uri=https%3A%2F%2Festraneo.example%2F" | grep -o 'Invalid parameter: redirect_uri' | head -1)"
  prova "il worker CPU annuncia le sue capacità" '"role":"cpu"' \
    "$(curl -s --max-time 15 "http://api.personalities.localtest.me:$PORTA/capabilities")"

  echo "Mesh"
  local senza
  senza="$(k -n personalities get pods -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.initContainers[*].name}{"\n"}{end}' \
    | grep -v istio-proxy | grep -v '^$' || true)"
  if [[ -z "$senza" ]]; then
    echo "  ok   ogni pod ha il proxy, come sidecar nativo"
  else
    echo "  NO   pod senza proxy: $senza"
    falliti=$((falliti + 1))
  fi
  # Lo schema all'ultima revisione, e non «Running upgrade» nei log: un pod
  # partito dopo il primo trova le migrazioni già applicate e non ne scrive.
  prova "schema del database all'ultima revisione" "(head)" \
    "$(k -n personalities exec deploy/api -c api -- python -m alembic current 2>&1 | tail -1)"
  # mTLS STRICT: un pod fuori dalla mesh non raggiunge l'API.
  k create namespace fuori-mesh --dry-run=client -o yaml | k apply -f - >/dev/null
  local fuori
  fuori="$(k -n fuori-mesh run prova-mtls --rm -i --restart=Never --image=curlimages/curl:8.11.1 --quiet -- \
    curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://api.personalities.svc.cluster.local:8100/healthz 2>/dev/null || true)"
  if [[ "$fuori" == "200" ]]; then
    echo "  NO   mTLS: un pod fuori dalla mesh ha raggiunto l'API"
    falliti=$((falliti + 1))
  else
    echo "  ok   mTLS: dal fuori-mesh l'API non risponde ($fuori)"
  fi

  echo "Avvio in produzione"
  prova "l'API parte con environment=prod" "Avvio di persona-api" "$(k -n personalities logs deploy/api -c api 2>&1 | head -c 20000)"
  # Un Job nella mesh deve finire: con i sidecar classici il proxy resterebbe
  # vivo e il Job non si chiuderebbe mai.
  k -n personalities delete job prova-manutenzione --ignore-not-found >/dev/null
  k -n personalities create job prova-manutenzione --from=cronjob/manutenzione >/dev/null
  prova "la manutenzione termina (sidecar nativo)" "condition met" \
    "$(k -n personalities wait --for=condition=complete job/prova-manutenzione --timeout=300s 2>&1)"
  k -n personalities delete job prova-manutenzione --ignore-not-found >/dev/null

  if (( falliti > 0 )); then
    echo "$falliti verifiche non superate."
    exit 1
  fi
  echo "Tutte le verifiche superate."
}

ferma() {
  docker rm -f "$NOME" >/dev/null 2>&1 || true
  rm -f "$KUBECONFIG_LOCALE" "$KUBECONFIG_INTERNO"
  echo "Cluster di prova rimosso."
}

case "${1:-}" in
  avvia) avvia ;;
  immagini) carica_immagini ;;
  istio) installa_istio ;;
  applica) applica ;;
  verifica) verifica ;;
  tutto) avvia; carica_immagini; installa_istio; applica; verifica ;;
  ferma) ferma ;;
  *) echo "uso: $0 {tutto|avvia|immagini|istio|applica|verifica|ferma}" >&2; exit 2 ;;
esac
