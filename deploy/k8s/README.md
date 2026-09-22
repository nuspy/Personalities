# Deployment su Kubernetes

La piattaforma su un cluster k3s (quello di `tadamoo.com`) dietro Istio, con
i manifest in kustomize. **Qui c'è la preparazione, non il rilascio**: si prova
in locale con `prova-locale.sh`, e si rilascia a mano con `rilascia.sh` quando
i segnaposto sono compilati.

## Cosa c'è

```
base/                  la piattaforma, senza ambiente
  api.yaml             API (2 repliche, HPA 2–8, PDB); migrazioni nell'initContainer
  worker-cpu.yaml      digestione e ingestione (+ battito delle capacità)
  worker-gpu.yaml      addestramenti — 0 repliche: senza GPU la piattaforma è RAG puro
  web.yaml, admin.yaml le due interfacce Next (standalone)
  keycloak.yaml        Keycloak 26 in modalità produzione, realm importato all'avvio
  postgres.yaml        PostgreSQL 17 + pgvector; Keycloak in un database suo
  redis.yaml           code e battiti, con AOF su disco
  manutenzione.yaml    CronJob ogni 10 minuti: rinnovi, memorie, consolidamento
  osservabilita.yaml   collector OpenTelemetry → Jaeger
  volumi.yaml          caricamenti (RWX fra API e worker), artefatti
istio/                 componente: gateway, instradamento, mTLS STRICT, autorizzazioni, tracce
  istio-installazione.yaml   l'installazione di Istio che i manifest si aspettano
overlays/
  produzione/          personalities.tadamoo.com; segreti da segreti.env e keycloak.env
  produzione/tls/      certificato Let's Encrypt (cert-manager), nello spazio di Istio
  locale/              la stessa cosa su un k3s dentro Docker, per provarla
prova-locale.sh        la prova in locale, dall'avvio del cluster alle verifiche
rilascia.sh            il rilascio in produzione (prova per default, --conferma per applicare)
```

Le immagini: `deploy/Dockerfile.backend` (stadi `api` e `worker`),
`deploy/Dockerfile.next` (`APP=frontend` o `APP=admin`),
`deploy/Dockerfile.worker-gpu`.

## Le scelte che contano

**Le migrazioni nell'initContainer dell'API**, in fila su un lock consultivo di
Postgres (`platform_core/migrations/env.py`): più repliche che partono insieme
non si pestano i piedi. Durante un rilascio la versione vecchia continua a
servire, quindi ogni migrazione deve restare compatibile con il codice
precedente — si aggiunge prima, si toglie in un rilascio successivo.

**In produzione l'API non parte con una configurazione insicura**: autenticazione
spenta, CORS aperto, database su localhost, pagamento simulato. Finché il
fornitore di pagamento vero non è scelto, `PERSONA_BILLING_PROVIDER=disattivato`:
i piani gratuiti funzionano, quelli a pagamento rispondono 503 col motivo.

**Istio con i sidecar nativi.** Con STRICT nessun servizio parla in chiaro, e
con i sidecar classici l'initContainer delle migrazioni partirebbe prima del
proxy e il CronJob non finirebbe mai. `istio-installazione.yaml` li attiva
(`ENABLE_NATIVE_SIDECARS`), e dichiara il fornitore di tracce `otel`.

**Nessun nuovo tentativo verso l'API**: una `POST /chat` ripetuta dopo che
l'API l'aveva ricevuta consumerebbe i crediti due volte. Nessun timeout sulle
risposte in streaming.

**Due indirizzi per Keycloak**: l'API legge le chiavi dal servizio interno
(`PERSONA_KEYCLOAK_URL=http://keycloak:8080`) e si aspetta nei token il dominio
pubblico (`PERSONA_KEYCLOAK_ISSUER_URL`), che Keycloak scrive grazie a
`KC_HOSTNAME`. La sua console di amministrazione non è esposta: si raggiunge
con `kubectl port-forward`.

**Gli indirizzi pubblici delle interfacce si decidono alla build** delle
immagini Next, non all'avvio: il browser non vede l'ambiente del container.

## Provare in locale

Serve Docker. Nessun modello viene interrogato: le verifiche guardano che i
servizi partano, rispondano sui loro domini e si parlino in mTLS.

```bash
# le immagini, con gli indirizzi dell'overlay locale
docker build -f deploy/Dockerfile.backend --target api    -t persona-api:locale .
docker build -f deploy/Dockerfile.backend --target worker -t persona-worker:locale .
docker build -f deploy/Dockerfile.next --build-arg APP=frontend \
  --build-arg NEXT_PUBLIC_API_URL=http://api.personalities.localtest.me:8088 \
  --build-arg NEXT_PUBLIC_KEYCLOAK_URL=http://auth.personalities.localtest.me:8088 \
  -t persona-web:locale .
docker build -f deploy/Dockerfile.next --build-arg APP=admin \
  --build-arg NEXT_PUBLIC_API_URL=http://api.personalities.localtest.me:8088 \
  --build-arg NEXT_PUBLIC_KEYCLOAK_URL=http://auth.personalities.localtest.me:8088 \
  -t persona-admin:locale .

deploy/k8s/prova-locale.sh tutto     # k3s in Docker, Istio, overlay locale, verifiche
deploy/k8s/prova-locale.sh ferma     # e via
```

Il cluster di prova espone il gateway sulla **8088** e la sua API sulla
**16443** (mai la 8000, che è di altri progetti), e usa un kubeconfig suo
(`deploy/k8s/.kube-locale.yaml`): lo script controlla che punti a 127.0.0.1
prima di applicare qualunque cosa. I domini `*.personalities.localtest.me`
risolvono a 127.0.0.1 senza toccare il file hosts.

## Prima del primo rilascio

| Da decidere o compilare | Dove |
|---|---|
| il registro delle immagini | `REGISTRO` in `overlays/produzione/kustomization.yaml` |
| l'indirizzo del motore dei modelli (LM Studio / vLLM via Tailscale, o Anthropic) | `INDIRIZZO_TAILSCALE`, stesso file |
| la classe di storage ReadWriteMany per i caricamenti | `CLASSE_RWX`, stesso file |
| i segreti | `overlays/produzione/segreti.env` e `keycloak.env`, dagli `.example` |
| l'email per gli avvisi del certificato | `EMAIL_AVVISI_CERTIFICATI` in `overlays/produzione/tls/certificato.yaml` |
| i DNS dei quattro domini verso l'IP del gateway | presso il registrar di `tadamoo.com` |
| cert-manager e Istio nel cluster | `istioctl install -f deploy/k8s/istio/istio-installazione.yaml` |

Poi:

```bash
kubectl --context <contesto> apply -k deploy/k8s/overlays/produzione/tls
deploy/k8s/rilascia.sh --contesto <contesto> --versione 0.1.0            # prova e diff
deploy/k8s/rilascia.sh --contesto <contesto> --versione 0.1.0 --conferma # rilascio
```

Dopo il primo avvio: il catalogo dei piani (`python -m platform_core.tools.seed_piani`
in un pod dell'API) e il primo amministratore in Keycloak, che al primo accesso
configura l'OTP.

## Accensione del motore locale

Un modello su GPU occupa la scheda anche mentre nessuno lo interroga. Il
worker può accenderlo quando gli serve e spegnerlo dopo un po' che nessuno lo
usa — **facoltativo**: senza configurazione il modello si gestisce a mano e
tutto funziona come prima.

Come è fatto, e perché così:

- **i comandi stanno nei segreti, non nell'interfaccia.** Sono righe eseguite
  dal worker con i suoi diritti: se una pagina potesse cambiarle, quella
  pagina sarebbe esecuzione di codice arbitrario. La console può chiedere
  «accendi» o «spegni»; non può dire *cosa* eseguire;
- **accende il worker, non l'API.** Il processo esposto al pubblico non ha il
  client SSH nella sua immagine e non raggiunge la macchina dei modelli:
  deposita una parola in Redis, che il worker raccoglie entro pochi secondi;
- **la chiave è a comando forzato.** Sulla macchina dei modelli,
  `authorized_keys` porta una riga per l'accensione e una per lo spegnimento:

  ```
  command="powershell -ExecutionPolicy Bypass -File C:\\Projects\\bonsai_2_server\\start.ps1",
  no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty,
  from="<IP/32 di egress del cluster>" ssh-ed25519 AAAA… motore-accendi
  ```

  Qualunque cosa il worker mandi, quella chiave esegue solo quel comando, e
  solo da quell'indirizzo. È ciò che rende accettabile tenere una chiave SSH
  in un pod;
- **le chiavi si generano al dispiegamento**, non prima: la riga `from=` vuole
  l'indirizzo di egress vero dei pod, che dipende da come il tailnet è
  attestato sul nodo e si legge sul cluster vivo. Le private restano nel
  Secret `motore-ssh`, le pubbliche si installano sulla macchina dei modelli.

Da compilare in `segreti.env` (vedi l'esempio): `PERSONA_LOCAL_ENGINE_START`,
`_STOP`, `_HEALTH_URL`, `_WAIT_S`, `_IDLE_S`. Poi si scommenta il
`secretGenerator` `motore-ssh` nell'overlay e si mettono i tre file in
`overlays/produzione/motore-ssh/` (`accendi`, `spegni`, `known_hosts`).

Lo stato si guarda dalla console, in *Digestione*: chi lo gestisce, da quanto
è acceso, quanti lavori lo stanno usando — e lì stanno i due pulsanti. Uno
stato che nessun worker aggiorna scade da solo e diventa «non gestito», invece
di restare un «acceso» che nessuno smentisce.

## Crescere

- **API e sito** scalano da soli sulla CPU (HPA). Sono senza stato.
- **Worker CPU**: una replica in più per ogni digestione o ingestione che si
  vuole in parallelo; per scalare sulla lunghezza della coda serve KEDA, con
  uno scaler Redis su `builds:in_coda:cpu`.
- **Worker GPU**: `replicas: 1` su un nodo con driver NVIDIA e device plugin.
- **Postgres** è il punto singolo: per l'alta disponibilità, un operatore
  (CloudNativePG) al posto dello StatefulSet, con i backup continui.
