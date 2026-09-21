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
  const [dati, setDati] = useState<T | null>(null);
  const [errore, setErrore] = useState<string | null>(null);
  const [inCorso, setInCorso] = useState(true);
  const [quando, setQuando] = useState(0);

  const ricarica = useCallback(() => setQuando((n) => n + 1), []);

  useEffect(() => {
    if (!token) return;

    let annullato = false;
    setInCorso(true);
    setErrore(null);

    fetcher(token)
      .then((risultato) => {
        if (!annullato) setDati(risultato);
      })
      .catch((e: unknown) => {
        if (annullato) return;
        setErrore(
          e instanceof ErroreApi ? e.message : "Qualcosa non ha funzionato.",
        );
      })
      .finally(() => {
        if (!annullato) setInCorso(false);
      });

    return () => {
      /* Una risposta che arriva dopo che il componente è sparito — o dopo che
       * i parametri sono cambiati — non deve scrivere sullo stato: metterebbe
       * a schermo il risultato di una richiesta che non è più quella in
       * corso. */
      annullato = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, quando, ...dipendenze]);

  return { dati, errore, inCorso, ricarica };
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
