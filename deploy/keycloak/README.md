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

Google, Facebook e Apple si aggiungono qui sotto `identityProviders` quando ci
saranno le credenziali. La struttura li accetta senza modifiche al codice: la
verifica del token guarda l'emittente e il destinatario, non da quale
provider l'utente sia arrivato.

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
