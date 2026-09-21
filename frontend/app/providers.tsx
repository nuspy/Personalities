"use client";

import { useEffect, useMemo, useState } from "react";
import { AuthProvider } from "react-oidc-context";
import { WebStorageStateStore } from "oidc-client-ts";
import type { User } from "oidc-client-ts";

/* Autenticazione: il codice di autorizzazione con PKCE.
 *
 * Il client è pubblico — gira in un browser, dove nessun segreto resta
 * segreto — quindi PKCE non è un'opzione fra le altre: è ciò che impedisce a
 * chi intercettasse il codice di riscattarlo.
 *
 * I token stanno in `sessionStorage` e non in `localStorage`: durano quanto la
 * scheda, e un token che sopravvive alla chiusura del browser su una macchina
 * condivisa è un token che qualcun altro può usare. Il costo è un nuovo login
 * aprendo una scheda nuova, e la sessione SSO di Keycloak lo rende indolore.
 *
 * **Perché il provider si monta solo dopo l'idratazione.** `redirect_uri` si
 * ricava da `window.location.origin`, che sul server non esiste. Montandolo
 * subito, l'`UserManager` nascerebbe con un indirizzo di ritorno vuoto e lo
 * terrebbe per sempre: il pulsante d'accesso non farebbe nulla — senza errori
 * in console, che è il modo peggiore in cui un difetto possa presentarsi.
 */

const authority = `${
  process.env.NEXT_PUBLIC_KEYCLOAK_URL ?? "http://localhost:8080"
}/realms/${process.env.NEXT_PUBLIC_KEYCLOAK_REALM ?? "personalities"}`;

function onSigninCallback(_user: User | void): void {
  /* Toglie `code` e `state` dalla barra degli indirizzi dopo il login.
   * Senza, un aggiornamento della pagina rimanderebbe a Keycloak un codice già
   * consumato, e l'errore che ne esce parla di `invalid_grant` — cioè sembra
   * un guasto dell'autenticazione e non un semplice ricaricamento. */
  window.history.replaceState({}, document.title, window.location.pathname);
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [montato, setMontato] = useState(false);

  useEffect(() => setMontato(true), []);

  const config = useMemo(
    () =>
      montato
        ? {
            authority,
            client_id:
              process.env.NEXT_PUBLIC_KEYCLOAK_CLIENT_ID ?? "persona-frontend",
            redirect_uri: window.location.origin,
            post_logout_redirect_uri: window.location.origin,
            response_type: "code",
            scope: "openid profile email",
            /* Rinnovo silenzioso prima della scadenza: senza, una
             * conversazione lunga si interrompe con un 401 a metà di una
             * risposta. */
            automaticSilentRenew: true,
            userStore: new WebStorageStateStore({
              store: window.sessionStorage,
            }),
            onSigninCallback,
          }
        : null,
    [montato],
  );

  if (!config) return null;

  return <AuthProvider {...config}>{children}</AuthProvider>;
}
