# Console di amministrazione

Next.js 16, porta 3001. Gestione di personalità, corpora, tassonomia e
registro.

## Avvio

```bash
npm install
npm run dev -- --port 3001
```

Serve il backend in ascolto e Keycloak avviato, con un account che porta il
ruolo `admin` — nel realm di sviluppo è `admin@example.com`.

## Perché un'applicazione separata dal frontend

Stessa palette e stessa tipografia d'identità, registro diverso. La chat è una
pagina di lettura: misura larga, molto respiro, un'informazione per volta. Qui
si amministra, e chi amministra guarda molte righe insieme per confrontarle —
densità maggiore, tabelle, caratteri più piccoli.

Separarle significa anche che un token della console non vale per
l'applicazione utente: sono due client OIDC distinti, con redirect propri.

I token del sistema visivo sono **copiati** da `frontend/app/globals.css` e non
condivisi: con due applicazioni un pacchetto comune costa più di quanto renda.
Alla terza varrà la pena estrarlo.

## Le pagine

| | |
|---|---|
| `/` | catalogo delle personalità, creazione |
| `/personalita/[id]` | versioni, corpora, tipi, scheda Realizzazione |
| `/corpora` | basi di conoscenza e loro documenti |
| `/recupero` | interroga il recupero **senza generare** |
| `/registro` | chi ha fatto cosa, e com'era prima |

## `/recupero` è la pagina che vale di più

Quando una risposta è sbagliata la domanda è sempre la stessa: il recupero non
ha trovato ciò che serviva, o l'ha trovato e scartato? Sono due difetti con due
rimedi opposti — nel primo caso si lavora sul corpus, nel secondo
sull'ordinamento — e distinguerli richiede di vedere il recupero senza la
generazione in mezzo. Qui si vedono i punteggi di entrambe le ricerche, la
posizione in ciascuna, e ciò che è rimasto fuori.

## Il ruolo

Il controllo nel guscio è **cortesia**: la sicurezza sta nel backend, che
risponde 403 a chiunque non porti `admin` a prescindere da cosa mostri
l'interfaccia. Serve a non far vedere a un utente comune una console piena di
pulsanti che falliranno tutti.
