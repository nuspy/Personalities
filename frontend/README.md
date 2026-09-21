# Frontend

Next.js 16 (App Router, React 19). Conversazione in streaming, accesso da
Keycloak.

## Avvio

```bash
npm install
npm run dev        # http://localhost:3000
```

Serve il backend in ascolto e Keycloak avviato:

```bash
docker compose up -d postgres redis keycloak     # dalla radice del progetto
python -m platform_core.api --port 8000
```

La configurazione sta in `.env.local` (copia `.env.example`). I valori sono
tutti `NEXT_PUBLIC_*` perché servono al browser: qui non ci sono segreti da
custodire — il client OIDC è pubblico, e la sua sicurezza sta in PKCE, non in
una chiave nascosta in un file.

## Perché non è nel docker-compose

Lo stack di sviluppo porta su database, Redis, Keycloak, tracce, API e worker.
Il frontend no, per la stessa ragione del worker con GPU: in sviluppo si
lavora con `next dev`, dove il ricaricamento è immediato, mentre dentro un
container su Windows il rilevamento delle modifiche passa da un file system
montato ed è lento al punto da cambiare il modo di lavorare. Nel deploy vero
diventa un'immagine come le altre.

## Struttura

| | |
|---|---|
| `app/providers.tsx` | client OIDC; si monta dopo l'idratazione, vedi il commento dentro |
| `app/page.tsx` | accesso e conversazione |
| `app/globals.css` | il sistema visivo: carta, inchiostro, rubrica |
| `lib/api.ts` | client del backend e lettura del flusso SSE |

## La direzione visiva

Il vocabolario è quello di un'edizione annotata: una colonna di lettura e, a
margine, l'apparato. Oggi quel margine porta chi parla e l'ora; dalla fase 1
ospiterà i riferimenti numerati ai passaggi recuperati, che è la ragione per
cui esiste fin d'ora.

Il rosso di rubrica marca i segni tipografici e non le azioni: un pulsante
rosso attirerebbe l'occhio su di sé, mentre la rubrica serve a dire dove
comincia una voce.

Due vincoli verificati e da non perdere:

- ogni coppia testo/sfondo sta sopra 4,5:1 in entrambi i temi — il testo
  dell'apparato è il più piccolo della pagina ed è quello che per primo scende
  sotto soglia;
- il campo di scrittura è a 16px esatti: sotto quella misura iOS ingrandisce
  la pagina a ogni tocco sul campo.
