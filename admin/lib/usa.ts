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
 * Carica dati dall'API e tiene lo stato dei tre casi che una pagina deve
 * saper mostrare: in corso, riuscito, fallito.
 *
 * I tre sono distinti perché «nessun dato» e «non ancora caricato» sembrano
 * uguali sullo schermo e sono diversi: il primo merita un invito ad agire, il
 * secondo un'attesa. Mostrarli allo stesso modo è il motivo per cui certe
 * pagine sembrano vuote quando invece stanno lavorando.
 */
export function useDati<T>(
  fetcher: (token: string) => Promise<T>,
  dipendenze: unknown[] = [],
): Caricamento<T> {
  const token = useToken();
  const [quando, setQuando] = useState(0);
  const ricarica = useCallback(() => setQuando((n) => n + 1), []);

  /* Che cosa decide se ricaricare.
   *
   * Le dipendenze esplicite quando ci sono; altrimenti l'identità della
   * funzione. La seconda regola è quella che mancava: una pagina che passa un
   * `useCallback` con i suoi filtri e nessuna dipendenza esplicita non
   * ricaricava mai al cambiare di un filtro, e il clic sembrava non fare
   * niente. Con una funzione definita al volo senza dipendenze l'identità
   * cambierebbe a ogni rendering: chi la usa così deve passarle. */
  const chiave: unknown[] = [token, quando, ...(dipendenze.length ? dipendenze : [fetcher])];

  const [esito, setEsito] = useState<{
    chiave: unknown[];
    dati: T | null;
    errore: string | null;
  } | null>(null);

  useEffect(() => {
    if (!token) return;
    let annullato = false;

    /* Lo stato si scrive solo quando la risposta arriva, mai all'inizio:
     * «in corso» si deduce confrontando la chiave dell'ultimo esito con
     * quella attuale. Impostarlo qui dentro in modo sincrono costringerebbe
     * React a un secondo rendering per dire una cosa già deducibile. */
    fetcher(token)
      .then((risultato) => {
        if (!annullato) setEsito({ chiave, dati: risultato, errore: null });
      })
      .catch((e: unknown) => {
        if (annullato) return;
        const messaggio =
          e instanceof ErroreApi ? e.message : "Qualcosa non ha funzionato.";
        /* I dati di prima restano: un ricaricamento fallito non deve
         * svuotare la pagina di ciò che si stava guardando. */
        setEsito((prima) => ({ chiave, dati: prima?.dati ?? null, errore: messaggio }));
      });

    return () => {
      /* Una risposta che arriva dopo che il componente è sparito — o dopo che
       * i parametri sono cambiati — non deve scrivere sullo stato: metterebbe
       * a schermo il risultato di una richiesta che non è più quella in
       * corso. */
      annullato = true;
    };
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
        setErrore(
          e instanceof ErroreApi ? e.message : "L'operazione non è riuscita.",
        );
        return null;
      } finally {
        setInCorso(false);
      }
    },
    [token],
  );

  return { esegui, inCorso, errore };
}
