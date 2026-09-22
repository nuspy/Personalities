"use client";

import { useEffect } from "react";
import { comandaMotore, statoMotore, type StatoMotore } from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import stili from "./motore.module.css";

/* Il modello locale: acceso, spento, e chi lo sta usando.
 *
 * La console **chiede**, non esegue: i comandi di accensione stanno nella
 * configurazione del worker, e nessuna pagina può cambiarli. Il pulsante
 * manda una parola — accendi, spegni — e lo stato qui cambia quando è
 * cambiato davvero, non quando è stato chiesto. È la ragione per cui dopo un
 * clic compare «richiesta inviata» e non «acceso».
 *
 * Lo stato si ricarica da solo: lo pubblica un worker con una scadenza, e una
 * pagina ferma mostrerebbe per minuti un motore che nel frattempo si è
 * spento da sé.
 */

const OGNI_MS = 10_000;

const DESCRIZIONI: Record<StatoMotore["stato"], { testo: string; segno: string }> = {
  non_gestito: { testo: "non gestito", segno: "—" },
  spento: { testo: "spento", segno: "○" },
  in_accensione: { testo: "in accensione", segno: "◐" },
  acceso: { testo: "acceso", segno: "●" },
  in_spegnimento: { testo: "in spegnimento", segno: "◑" },
  guasto: { testo: "guasto", segno: "!" },
};

export function Motore() {
  const { dati, errore, ricarica } = useDati(statoMotore);
  const { esegui, inCorso, errore: erroreAzione } = useAzione();

  useEffect(() => {
    const t = setInterval(ricarica, OGNI_MS);
    return () => clearInterval(t);
  }, [ricarica]);

  if (errore) return <p className={stili.errore}>{errore}</p>;
  if (!dati) return null;

  const d = DESCRIZIONI[dati.stato] ?? DESCRIZIONI.non_gestito;
  const gestito = dati.stato !== "non_gestito";
  const occupato = dati.stato === "in_accensione" || dati.stato === "in_spegnimento";

  const chiedi = async (azione: "accendi" | "spegni") => {
    if (await esegui((t) => comandaMotore(t, azione))) ricarica();
  };

  return (
    <section className={stili.pannello} aria-label="Motore locale">
      <div className={stili.riga}>
        <h2 className={stili.titolo}>Modello locale</h2>
        {/* Segno e parola insieme: il colore da solo non dice niente a chi
            non lo distingue, e un pallino verde non ha un nome. */}
        <span className={`${stili.stato} ${stili[dati.stato]}`}>
          <span aria-hidden="true">{d.segno}</span> {d.testo}
        </span>
      </div>

      <p className={stili.nota}>
        {dati.stato === "acceso" && dati.in_corso > 0 && (
          <>
            {dati.in_corso === 1
              ? "Un lavoro lo sta usando"
              : `${dati.in_corso} lavori lo stanno usando`}
            : non si spegne finché non hanno finito.{" "}
          </>
        )}
        {dati.stato === "acceso" && dati.in_corso === 0 && dati.inattivita_s > 0 && (
          <>Si spegne da solo dopo {minuti(dati.inattivita_s)} senza lavori. </>
        )}
        {dati.motivo}
      </p>

      {gestito && (
        <div className={stili.comandi}>
          <button
            className={stili.comando}
            disabled={inCorso || occupato || dati.stato === "acceso"}
            onClick={() => chiedi("accendi")}
          >
            Accendi
          </button>
          <button
            className={stili.comando}
            disabled={inCorso || occupato || dati.stato === "spento"}
            onClick={() => chiedi("spegni")}
          >
            Spegni
          </button>
          {dati.worker_id && (
            <span className={stili.chi}>gestito da {dati.worker_id}</span>
          )}
        </div>
      )}

      {erroreAzione && <p className={stili.errore}>{erroreAzione}</p>}
    </section>
  );
}

function minuti(secondi: number): string {
  if (secondi < 90) return `${Math.round(secondi)} secondi`;
  const m = Math.round(secondi / 60);
  return m === 1 ? "un minuto" : `${m} minuti`;
}
