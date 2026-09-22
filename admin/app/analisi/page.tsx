"use client";

/* Come sta andando la piattaforma.
 *
 * Prima i numeri, poi l'andamento, poi la ripartizione: chi apre questa
 * pagina vuole sapere se qualcosa è cambiato, e un grafico prima dei numeri
 * lo costringe a leggere un'area per ricavare ciò che una cifra dice subito.
 * Ogni numero porta il confronto col periodo precedente di pari durata.
 */

import { useCallback, useState } from "react";
import { leggiAnalisi } from "@/lib/api";
import { useDati } from "@/lib/usa";
import comuni from "../comuni.module.css";
import { Barre, Colonne, Tessera, compatto, percento } from "../grafici";
import grafici from "../grafici.module.css";
import stili from "./analisi.module.css";

const PERIODI = [7, 30, 90];

const giornoBreve = new Intl.DateTimeFormat("it-IT", { day: "numeric", month: "short" });

/** Sotto il secondo in millisecondi, sopra in secondi: «14,9k ms» non lo
 *  legge nessuno. */
function durata(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1).replace(".", ",")} s`;
}

function quanti(n: number, uno: string, molti: string): string {
  return `${compatto(n)} ${n === 1 ? uno : molti}`;
}

export default function Analisi() {
  const [giorni, setGiorni] = useState(30);
  const { dati, errore, inCorso } = useDati(
    useCallback((t: string) => leggiAnalisi(t, giorni), [giorni]),
  );

  const ora = dati?.indicatori;
  const prima = dati?.precedente;

  return (
    <>
      <div className={comuni.intestazione}>
        <div>
          <h1 className={comuni.titolo}>Analisi</h1>
          <p className={comuni.sottotitolo}>
            Uso, gradimento e qualità delle risposte. Ogni numero si calcola da
            ciò che la piattaforma registra — messaggi, tracce, voti, crediti —
            e porta accanto il periodo precedente di pari durata.
          </p>
        </div>
      </div>

      {/* I filtri sopra tutto ciò che filtrano, in una riga sola. */}
      <div className={stili.filtri} role="group" aria-label="Periodo">
        {PERIODI.map((g) => (
          <button
            key={g}
            className={g === giorni ? stili.periodoScelto : stili.periodo}
            aria-pressed={g === giorni}
            onClick={() => setGiorni(g)}
          >
            Ultimi {g} giorni
          </button>
        ))}
      </div>

      {errore && <p className={comuni.errore}>{errore}</p>}
      {!dati && inCorso && <p className={comuni.caricamento}>Caricamento…</p>}

      {ora && prima && dati && (
        /* Mentre si ricarica, il quadro resta: più chiaro, non vuoto. */
        <div className={inCorso ? stili.inRicarica : undefined}>
          <div className={grafici.tessere}>
            <Tessera etichetta="Risposte" valore={ora.risposte} precedente={prima.risposte} />
            <Tessera etichetta="Utenti attivi" valore={ora.utenti_attivi} precedente={prima.utenti_attivi} />
            <Tessera etichetta="Crediti consumati" valore={ora.crediti} precedente={prima.crediti} />
            <Tessera
              etichetta="Approvazione"
              valore={ora.voti.approvazione}
              precedente={prima.voti.approvazione}
              formato={(n) => percento(n)}
              nota={`${quanti(ora.voti.su, "utile", "utili")}, ${quanti(ora.voti.giu, "non utile", "non utili")}`}
            />
            <Tessera
              etichetta="Citazioni inventate"
              valore={ora.citazioni_inventate}
              precedente={prima.citazioni_inventate}
              formato={(n) => percento(n, 1)}
              meglio="giu"
              nota="risposte che citano un passaggio non fornito"
            />
            <Tessera
              etichetta="Primo token, mediana"
              valore={ora.primo_token_ms.p50}
              precedente={prima.primo_token_ms.p50}
              formato={durata}
              meglio="giu"
              nota={ora.primo_token_ms.p95 !== null ? `p95 ${durata(ora.primo_token_ms.p95)}` : undefined}
            />
            <Tessera
              etichetta="Prompt letto da cache"
              valore={ora.cache.quota}
              precedente={prima.cache.quota}
              formato={(n) => percento(n)}
              nota={`${compatto(ora.cache.token_da_cache)} di ${compatto(ora.cache.token_prompt)} token`}
            />
          </div>

          <Colonne
            titolo="Risposte al giorno"
            descrizione={`Dal ${giornoBreve.format(new Date(dati.periodo.dal))}, giorni senza attività compresi.`}
            unita="risposte"
            punti={dati.al_giorno.map((g) => ({
              chiave: g.giorno,
              etichetta: giornoBreve.format(new Date(`${g.giorno}T12:00:00Z`)),
              valore: g.risposte,
            }))}
          />

          <Barre
            titolo="Risposte per personalità"
            descrizione="Nel periodo scelto; le conversazioni senza personalità non compaiono qui. La tabella aggiunge utenti e voti."
            unita="risposte"
            barre={dati.per_personalita.map((p) => ({
              chiave: p.slug,
              etichetta: p.nome,
              valore: p.risposte,
              dettaglio: `${quanti(p.utenti, "utente", "utenti")} · approvazione ${percento(p.approvazione)} (${quanti(p.su + p.giu, "voto", "voti")})`,
            }))}
            colonneTabella={[
              { titolo: "Utenti", valori: dati.per_personalita.map((p) => String(p.utenti)) },
              { titolo: "Utili", valori: dati.per_personalita.map((p) => String(p.su)) },
              { titolo: "Non utili", valori: dati.per_personalita.map((p) => String(p.giu)) },
              { titolo: "Approvazione", valori: dati.per_personalita.map((p) => percento(p.approvazione)) },
            ]}
          />

          {ora.verifica.verificate > 0 && (
            <p className={stili.nota}>
              Il verificatore ha giudicato {compatto(ora.verifica.verificate)} risposte:
              fondate {percento(ora.verifica.quota)}.
            </p>
          )}
        </div>
      )}
    </>
  );
}
