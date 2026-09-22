"use client";

/* Il voto su una risposta: utile, o no.
 *
 * Due pulsanti e niente scala: nessuno sa dire se una risposta valga tre o
 * quattro, e la media di numeri inventati è un numero inventato. Il motivo si
 * chiede solo a chi dice «no», e resta facoltativo: chiederlo sempre farebbe
 * smettere di votare. Un secondo tocco sullo stesso pulsante toglie il voto.
 *
 * I voti sono la misura con cui la console confronta due versioni della
 * stessa personalità: per questo il pulsante non promette altro che «serve a
 * migliorarla».
 */

import { useState } from "react";
import { ErroreApi, togliVoto, votaRisposta } from "@/lib/api";
import { useToken } from "@/lib/usa";
import stili from "./voto.module.css";

export function Voto({ messageId, iniziale }: { messageId: string; iniziale?: number | null }) {
  const token = useToken();
  const [voto, setVoto] = useState<number | null>(iniziale ?? null);
  const [motivo, setMotivo] = useState("");
  const [chiedeMotivo, setChiedeMotivo] = useState(false);
  const [inviato, setInviato] = useState(false);
  const [errore, setErrore] = useState<string | null>(null);

  const scegli = async (nuovo: 1 | -1) => {
    if (!token) return;
    const precedente = voto;
    const togli = voto === nuovo;
    // Il pulsante risponde subito; se il servizio rifiuta, si torna indietro
    // e si dice perché.
    setVoto(togli ? null : nuovo);
    setChiedeMotivo(!togli && nuovo === -1);
    setInviato(false);
    setErrore(null);
    try {
      if (togli) await togliVoto(token, messageId);
      else await votaRisposta(token, messageId, nuovo);
    } catch (e) {
      setVoto(precedente);
      setChiedeMotivo(false);
      setErrore(e instanceof ErroreApi ? e.message : "Il voto non è stato registrato.");
    }
  };

  const mandaMotivo = async () => {
    if (!token || !motivo.trim()) return;
    try {
      await votaRisposta(token, messageId, -1, motivo.trim());
      setChiedeMotivo(false);
      setInviato(true);
    } catch (e) {
      setErrore(e instanceof ErroreApi ? e.message : "Il motivo non è stato registrato.");
    }
  };

  return (
    <div className={stili.voto}>
      <div className={stili.pulsanti} role="group" aria-label="Com'era la risposta">
        <button
          type="button"
          className={voto === 1 ? stili.scelto : stili.pulsante}
          aria-pressed={voto === 1}
          aria-label="Risposta utile"
          title="Utile"
          onClick={() => scegli(1)}
        >
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <path d="M5 8v8H2V8h3Zm2 8h6.4a2 2 0 0 0 2-1.6l1-5A2 2 0 0 0 14.4 7H11V3.5A1.5 1.5 0 0 0 9.5 2L7 8v8Z"
              fill={voto === 1 ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
          </svg>
        </button>
        <button
          type="button"
          className={voto === -1 ? stili.scelto : stili.pulsante}
          aria-pressed={voto === -1}
          aria-label="Risposta non utile"
          title="Non utile"
          onClick={() => scegli(-1)}
        >
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <path d="M13 10V2h3v8h-3Zm-2-8H4.6a2 2 0 0 0-2 1.6l-1 5A2 2 0 0 0 3.6 11H7v3.5A1.5 1.5 0 0 0 8.5 16L11 10V2Z"
              fill={voto === -1 ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      {chiedeMotivo && (
        <form
          className={stili.motivo}
          onSubmit={(e) => {
            e.preventDefault();
            mandaMotivo();
          }}
        >
          <label htmlFor={`motivo-${messageId}`} className="solo-lettori">
            Cosa non andava
          </label>
          <input
            id={`motivo-${messageId}`}
            className={stili.campo}
            value={motivo}
            onChange={(e) => setMotivo(e.target.value)}
            placeholder="Cosa non andava? (facoltativo)"
            maxLength={1000}
          />
          <button className={stili.invia} disabled={!motivo.trim()}>
            Invia
          </button>
        </form>
      )}
      {inviato && <span className={stili.grazie}>Grazie: serve a migliorare questa voce.</span>}
      {errore && <span className={stili.errore}>{errore}</span>}
    </div>
  );
}
