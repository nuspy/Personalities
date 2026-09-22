# Realm di sviluppo

`realm-personalities.json` viene importato all'avvio di Keycloak (`start-dev
--import-realm`). Contiene i due ruoli della specifica, i client di frontend,
console e API, e due utenti di prova.

**Non usare questo file in produzione.** Le password sono note e il segreto del
client `persona-api` è un segnaposto.

## Perché non ci sono commenti nel JSON

Keycloak rifiuta le proprietà che non riconosce: una chiave `_commento` fa
fallire l'import con `Unrecognized field`, e il container resta in riavvio
continuo. Le note stanno qui.

## Contenuto

| Elemento | Nota |
|---|---|
| `user`, `admin` | i due ruoli di realm della specifica |
| `default-roles-personalities` | composito, assegna `user` a ogni nuovo iscritto |
| `persona-frontend` | client pubblico, PKCE S256, redirect su :3000 |
| `persona-admin` | console di amministrazione, :3001 |
| `persona-api` | client confidenziale, solo service account |
| `utente@example.com` / `utente` | utente semplice |
| `admin@example.com` / `admin` | utente con ruolo `admin` |

## Identity provider

Nel realm di sviluppo non ce ne sono: l'accesso locale usa gli utenti di
prova. Quelli veri stanno nel realm di produzione, qui sotto. Il codice non
cambia in nessun caso: la verifica del token guarda l'emittente e il
destinatario, non da quale provider l'utente sia arrivato.

## Riapplicare il realm dopo una modifica

L'import avviene solo alla prima creazione del database interno. Per rileggerlo:

    docker compose down keycloak
    docker volume rm personalities_keycloak_data 2>/dev/null || true
    docker compose up -d keycloak

## `persona-test` e il pubblico del token

Il client dei test automatici porta un mappatore di *audience* verso
`persona-api`. Senza, l'API rifiuta i suoi token con «token emesso per
['persona-test']»: la verifica del pubblico è voluta — un token buono per
un client non deve valere per un altro — ma un client di prova che non
riesce a parlare con ciò che deve provare non serve a niente.

# Realm di produzione

`realm-personalities.prod.json` è lo stesso realm senza utenti di prova, senza
il client `persona-test` e senza indirizzi `localhost`, con in più:

| | |
|---|---|
| 2FA per gli amministratori | OTP obbligatorio per chi ha il ruolo `admin`, anche quando entra con un provider esterno |
| 2FA per gli utenti | facoltativa: chi la attiva dal proprio account la usa, gli altri no |
| provider social | Google, Facebook, Apple — **spenti** finché non li si accende dall'ambiente |
| protezione brute force | blocco progressivo dopo 8 tentativi, fino a 15 minuti |
| password | almeno 12 caratteri, diversa da nome ed email, non una delle ultime 3 |
| email verificata | obbligatoria alla registrazione: serve l'SMTP |
| HTTPS | richiesto per le richieste esterne (`sslRequired: external`) |

## Variabili d'ambiente

Keycloak sostituisce i segnaposto `${...}` all'import (`--import-realm`); il
file non contiene segreti.

| Variabile | A cosa serve |
|---|---|
| `KC_URL_WEB`, `KC_URL_ADMIN` | gli indirizzi del sito e della console, per il ritorno dal login (in Kubernetes li mette l'overlay) |
| `KC_API_CLIENT_SECRET` | segreto del client confidenziale `persona-api` |
| `KC_SMTP_HOST`, `KC_SMTP_PORT`, `KC_SMTP_FROM`, `KC_SMTP_USER`, `KC_SMTP_PASSWORD` | invio delle email di verifica e di recupero |
| `KC_GOOGLE_ENABLED`, `KC_GOOGLE_CLIENT_ID`, `KC_GOOGLE_CLIENT_SECRET` | accesso con Google |
| `KC_FACEBOOK_ENABLED`, `KC_FACEBOOK_CLIENT_ID`, `KC_FACEBOOK_CLIENT_SECRET` | accesso con Facebook |
| `KC_APPLE_ENABLED`, `KC_APPLE_CLIENT_ID`, `KC_APPLE_CLIENT_SECRET` | accesso con Apple |

`KC_<PROVIDER>_ENABLED` vale `false` se manca: un provider senza credenziali
non compare nella pagina di accesso. Per accenderlo si impostano le tre
variabili e si registra presso il provider l'indirizzo di ritorno:

    https://<dominio di Keycloak>/realms/personalities/broker/<google|facebook|apple>/endpoint

## Come funziona la 2FA

Il flusso del browser è `browser con 2FA amministratori`, copia di quello
predefinito con due modifiche:

1. il sottoflusso *Conditional OTP* — l'OTP per chi l'ha attivato — esclude
   gli amministratori, o a loro verrebbe chiesto due volte;
2. un sottoflusso nuovo, condizionato al ruolo `admin`, chiede sempre l'OTP.
   Un amministratore che non l'ha ancora configurato viene portato a farlo al
   primo accesso, con il QR code da leggere con un'app (Google Authenticator,
   Microsoft Authenticator, FreeOTP).

Il flusso `2FA amministratori dopo accesso esterno` è il *post login* dei tre
provider: senza, un amministratore che entra con Google salterebbe il secondo
fattore, perché il flusso del browser finisce al reindirizzamento.

## Apple

Apple non ha un provider integrato in Keycloak ed è configurato come OIDC
generico. Il suo `client_secret` non è una stringa fissa ma un JWT firmato
(ES256) con la chiave privata dell'account sviluppatore, valido **al più sei
mesi**: va rigenerato prima della scadenza e aggiornato in
`KC_APPLE_CLIENT_SECRET`, o l'accesso con Apple smette di funzionare senza
preavviso. L'alternativa è l'estensione comunitaria
*keycloak-apple-social-identity-provider*, che lo firma da sé.

## Rigenerare il file

Il file si genera, non si scrive a mano: se un realm dichiara flussi di
autenticazione Keycloak non crea più quelli predefiniti, e un file con il solo
flusso 2FA importerebbe un realm senza recupero della password né
registrazione. Lo script costruisce i flussi su un'istanza usa-e-getta ed
esporta il realm intero:

    docker run -d --name kc-genera -p 8181:8080 -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
        -e KC_BOOTSTRAP_ADMIN_PASSWORD=genera -e KC_DB=dev-mem \
        quay.io/keycloak/keycloak:26.0 start-dev
    python deploy/keycloak/strumenti/genera_realm_prod.py --password genera
    docker rm -f kc-genera

Va rifatto dopo ogni modifica al realm di sviluppo che debba arrivare in
produzione.

## Verifica manuale prima del primo avvio

1. Importato il file su un'istanza di prova, entrare nella console come
   amministratore: al primo accesso compare la configurazione dell'OTP.
2. Entrare come utente senza OTP: nessun secondo fattore richiesto.
3. Con `KC_GOOGLE_ENABLED=true` e le credenziali, la pagina di accesso mostra
   Google; senza, non lo mostra.
