"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "react-oidc-context";
import { ErroreApi } from "./api";

/** Il token per le chiamate, o `null` finché la sessione non c'è. */
export function useToken(): string | null {
  const auth = useAuth();
  return auth.user?.access_token ?? null;
}

interface Caricamento<T> {
  dati: T | null;
  errore: string | null;
  inCorso: boolean;
  ricarica: () => void;
}

/**
 * Carica dati dall'API distinguendo in corso, riuscito e fallito.
 *
 * «In corso» si deduce dalla chiave dell'ultimo esito invece di essere
 * impostato all'inizio dell'effetto: così lo stato si scrive solo quando la
 * risposta arriva, e React non rende la pagina due volte per dire una cosa
 * già deducibile. Stesso schema della console.
 */
export function useCarica<T>(
  fetcher: (token: string) => Promise<T>,
  dipendenze: unknown[] = [],
): Caricamento<T> {
  const token = useToken();
  const [quando, setQuando] = useState(0);
  const ricarica = useCallback(() => setQuando((n) => n + 1), []);
  const chiave: unknown[] = [token, quando, ...(dipendenze.length ? dipendenze : [fetcher])];

  const [esito, setEsito] = useState<{ chiave: unknown[]; dati: T | null; errore: string | null } | null>(null);

  useEffect(() => {
    if (!token) return;
    let annullato = false;
    fetcher(token)
      .then((r) => { if (!annullato) setEsito({ chiave, dati: r, errore: null }); })
      .catch((e: unknown) => {
        if (annullato) return;
        const messaggio = e instanceof ErroreApi ? e.message : "Qualcosa non ha funzionato.";
        setEsito((prima) => ({ chiave, dati: prima?.dati ?? null, errore: messaggio }));
      });
    return () => { annullato = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, chiave);

  const aggiornato =
    esito !== null &&
    esito.chiave.length === chiave.length &&
    esito.chiave.every((v, i) => Object.is(v, chiave[i]));

  return {
    dati: esito?.dati ?? null,
    errore: aggiornato ? esito.errore : null,
    inCorso: !aggiornato,
    ricarica,
  };
}

/** Esegue un'azione che modifica, riportando l'esito. */
export function useAzione() {
  const token = useToken();
  const [inCorso, setInCorso] = useState(false);
  const [errore, setErrore] = useState<string | null>(null);

  const esegui = useCallback(
    async <T,>(azione: (t: string) => Promise<T>): Promise<T | null> => {
      if (!token) return null;
      setInCorso(true);
      setErrore(null);
      try {
        return await azione(token);
      } catch (e: unknown) {
        setErrore(e instanceof ErroreApi ? e.message : "L'operazione non è riuscita.");
        return null;
      } finally {
        setInCorso(false);
      }
    },
    [token],
  );

  return { esegui, inCorso, errore };
}
