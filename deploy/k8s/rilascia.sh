#!/usr/bin/env bash
# Il rilascio in produzione. **Preparato, non eseguito**: prima va provato in
# locale (prova-locale.sh), e va lanciato a mano da chi ha accesso al cluster.
#
#   deploy/k8s/rilascia.sh --contesto <contesto kubectl> --versione 0.1.0
#       costruisce le immagini, genera i manifest, mostra `kubectl diff`: non
#       cambia niente né nel registro né nel cluster
#   deploy/k8s/rilascia.sh --contesto <contesto> --versione 0.1.0 --conferma
#       spinge le immagini, applica, attende che tutto sia pronto
#
# Si rifiuta di procedere se nei manifest restano segnaposto (REGISTRO,
# INDIRIZZO_TAILSCALE, CLASSE_RWX…), se mancano i file dei segreti, o se il
# contesto indicato non esiste: un rilascio a metà è peggio di nessuno.
set -euo pipefail

QUI="$(cd "$(dirname "$0")" && pwd)"
RADICE="$(cd "$QUI/../.." && pwd)"
OVERLAY="$QUI/overlays/produzione"

CONTESTO=""
VERSIONE=""
CONFERMA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --contesto) CONTESTO="$2"; shift 2 ;;
    --versione) VERSIONE="$2"; shift 2 ;;
    --conferma) CONFERMA=1; shift ;;
    *) echo "argomento sconosciuto: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CONTESTO" && -n "$VERSIONE" ]] || { echo "servono --contesto e --versione" >&2; exit 2; }

k() { kubectl --context "$CONTESTO" "$@"; }

# --- controlli ---------------------------------------------------------------

k config get-contexts "$CONTESTO" >/dev/null 2>&1 || { echo "contesto $CONTESTO inesistente" >&2; exit 1; }
for f in segreti.env keycloak.env; do
  [[ -s "$OVERLAY/$f" ]] || { echo "manca $OVERLAY/$f: copialo da $f.example e compilalo" >&2; exit 1; }
done
if grep -qE '^[A-Z_]+=$' "$OVERLAY/segreti.env"; then
  echo "segreti.env ha valori vuoti:" >&2
  grep -nE '^[A-Z_]+=$' "$OVERLAY/segreti.env" >&2
  echo "(lascia vuoti solo quelli che non usi, e toglili dal file)" >&2
  exit 1
fi

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
kubectl kustomize --load-restrictor LoadRestrictionsNone "$OVERLAY" > "$manifest"
if grep -nE 'REGISTRO|INDIRIZZO_TAILSCALE|CLASSE_RWX|EMAIL_AVVISI|DOMINIO_[A-Z]+|DOMINI_CORS' "$manifest"; then
  echo "Nei manifest restano segnaposto (sopra): compila overlays/produzione prima di rilasciare." >&2
  exit 1
fi

registro="$(grep -oE 'image: [^ ]+/personalities/api:' "$manifest" | head -1 | sed -E 's#image: (.+)/personalities/api:#\1#')"
leggi() { python - "$OVERLAY/domini.yaml" "$1" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["data"][sys.argv[2]])
PY
}
API_URL="$(leggi api_url)"
AUTH_URL="$(leggi auth_url)"

# --- immagini ----------------------------------------------------------------

cd "$RADICE"
docker build -f deploy/Dockerfile.backend --target api -t "$registro/personalities/api:$VERSIONE" .
docker build -f deploy/Dockerfile.backend --target worker -t "$registro/personalities/worker:$VERSIONE" .
# Gli indirizzi pubblici entrano nel JavaScript alla build: un'immagine per
# dominio (vedi deploy/Dockerfile.next).
for app in frontend:web admin:admin; do
  docker build -f deploy/Dockerfile.next --build-arg APP="${app%%:*}" \
    --build-arg NEXT_PUBLIC_API_URL="$API_URL" \
    --build-arg NEXT_PUBLIC_KEYCLOAK_URL="$AUTH_URL" \
    -t "$registro/personalities/${app##*:}:$VERSIONE" .
done
# Il worker GPU pesa gigabyte e serve solo dove c'è un acceleratore: si
# costruisce a parte, quando lo si accende.

if grep -q "newTag: \"$VERSIONE\"" "$OVERLAY/kustomization.yaml"; then :; else
  echo "Attenzione: overlays/produzione/kustomization.yaml non porta newTag \"$VERSIONE\"." >&2
  echo "Aggiornalo, così che il repository dica quale versione è in produzione." >&2
  exit 1
fi

echo "--- differenze rispetto al cluster $CONTESTO ---"
k diff -f "$manifest" || true

if (( CONFERMA == 0 )); then
  echo "Prova terminata: niente è stato spinto né applicato. Rilancia con --conferma."
  exit 0
fi

# --- rilascio ----------------------------------------------------------------

for immagine in api worker web admin; do
  docker push "$registro/personalities/$immagine:$VERSIONE"
done
k apply -f "$manifest"
for risorsa in statefulset/postgres statefulset/redis deploy/keycloak deploy/api deploy/web deploy/admin deploy/worker-cpu; do
  k -n personalities rollout status "$risorsa" --timeout=900s
done
echo "Rilascio $VERSIONE completato su $CONTESTO."
