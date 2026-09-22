"use client";

/* L'avanzamento di un lavoro accodato: barra, ultima riga, diario.
 *
 * Lo usano la digestione e il caricamento dei documenti, che sono lo stesso
 * genere di cosa — un lavoro lungo del worker, con messaggi che arrivano via
 * via — e due copie della stessa barra finirebbero per comportarsi in due
 * modi diversi.
 *
 * Il flusso si segue finché il componente è montato, e si chiude quando non
 * lo è più: una connessione lasciata appesa a un lavoro che nessuno guarda
 * resta aperta fino al timeout del proxy. Per seguire un lavoro nuovo il
 * genitore cambia `key`, e lo stato riparte da zero senza un effetto che lo
 * azzeri.
 */

import { useEffect, useRef, useState } from "react";
import { seguiLavoro } from "@/lib/api";
import { useToken } from "@/lib/usa";
import stili from "./avanzamento.module.css";

interface Riga {
  id: number;
  progress: number;
  message: string;
  level: string;
}

export function Avanzamento({
  lavoro,
  etichetta,
  suFine,
}: {
  lavoro: string;
  etichetta: string;
  /** Chiamata una volta: con la build come la restituisce il servizio, o
   *  con `interrotto: true` se si è chiuso il flusso e non il lavoro. */
  suFine?: (build: Record<string, unknown>) => void;
}) {
  const token = useToken();
  const [righe, setRighe] = useState<Riga[]>([]);
  const [finito, setFinito] = useState<string | null>(null);
  const diario = useRef<HTMLUListElement>(null);
  /* In un riferimento: cambiare la funzione a ogni rendering del genitore
   * non deve riaprire il flusso. */
  const allaFine = useRef(suFine);
  useEffect(() => {
    allaFine.current = suFine;
  }, [suFine]);

  useEffect(() => {
    if (!token) return;
    return seguiLavoro(token, lavoro, (evento, dato) => {
      if (evento === "progress") {
        setRighe((prima) => [
          ...prima.slice(-199),
          {
            id: Number(dato.id),
            progress: Number(dato.progress ?? 0),
            message: String(dato.message ?? ""),
            level: String(dato.level ?? "info"),
          },
        ]);
      } else if (evento === "done") {
        setFinito(String(dato.status ?? "conclusa"));
        allaFine.current?.(dato);
      } else if (evento === "errore" || evento === "timeout") {
        /* Il flusso si è chiuso, non il lavoro: il genitore riabilita i
         * comandi, e il lavoro si ritrova nella pagina che lo elenca. */
        setFinito(String(dato.message ?? "flusso interrotto"));
        allaFine.current?.({ ...dato, interrotto: true });
      }
    });
  }, [lavoro, token]);

  useEffect(() => {
    diario.current?.scrollTo({ top: diario.current.scrollHeight });
  }, [righe]);

  const ultima = righe[righe.length - 1];

  return (
    <div className={stili.avanzamento}>
      <div
        className={stili.traccia}
        role="progressbar"
        aria-valuenow={ultima?.progress ?? 0}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={etichetta}
      >
        <div className={stili.riempimento} style={{ width: `${ultima?.progress ?? 0}%` }} />
      </div>
      <p className={stili.riga} aria-live="polite">
        <span>{finito ?? ultima?.message ?? "In attesa di un worker…"}</span>
        <span>{ultima?.progress ?? 0}%</span>
      </p>

      {righe.length > 0 && (
        <ul className={stili.diario} ref={diario}>
          {righe.map((r) => (
            <li key={r.id} className={r.level === "errore" ? stili.diarioErrore : undefined}>
              <span className={stili.diarioPercento}>{r.progress}%</span>
              <span>{r.message}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
