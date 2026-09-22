"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "react-oidc-context";
import stili from "./guscio.module.css";

const SEZIONI = [
  { href: "/", etichetta: "Personalità" },
  { href: "/corpora", etichetta: "Corpora" },
  { href: "/digestione", etichetta: "Digestione" },
  { href: "/avatar", etichetta: "Avatar" },
  { href: "/modelli", etichetta: "Modelli" },
  { href: "/prove", etichetta: "Prove" },
  { href: "/esperimenti", etichetta: "Esperimenti" },
  { href: "/analisi", etichetta: "Analisi" },
  { href: "/recupero", etichetta: "Recupero" },
  { href: "/registro", etichetta: "Registro" },
];

/* Il telaio della console: accesso, navigazione, e il perimetro del ruolo.
 *
 * Il controllo del ruolo è qui e non su ogni pagina — ma è **cortesia**, non
 * sicurezza: quella sta nel backend, che risponde 403 a chiunque non abbia
 * `admin` a prescindere da cosa mostri l'interfaccia. Serve a non far vedere
 * a un utente comune una console piena di pulsanti che falliranno tutti.
 */
export function Guscio({ children }: { children: React.ReactNode }) {
  const auth = useAuth();
  const percorso = usePathname();

  if (auth.isLoading) {
    return <p className={stili.attesa}>Verifica della sessione…</p>;
  }

  if (!auth.isAuthenticated) {
    return (
      <div className={stili.soglia}>
        <div className={stili.sogliaColonna}>
          <h1 className={stili.sogliaTitolo}>Console</h1>
          <p className={stili.sogliaTesto}>
            Amministrazione delle personalità, dei corpora e del catalogo.
            Serve un account con ruolo di amministratore.
          </p>
          <button
            className={stili.entra}
            onClick={() => auth.signinRedirect().catch(() => undefined)}
          >
            Entra
          </button>
        </div>
      </div>
    );
  }

  const ruoli =
    (auth.user?.profile as { realm_access?: { roles?: string[] } } | undefined)
      ?.realm_access?.roles ?? [];
  const amministratore = ruoli.includes("admin");

  return (
    <div className={stili.telaio}>
      <header className={stili.testata}>
        <Link href="/" className={stili.marchio}>
          Console<span>.</span>
        </Link>

        {amministratore && (
          <nav className={stili.navigazione}>
            {SEZIONI.map((s) => (
              <Link
                key={s.href}
                href={s.href}
                className={
                  percorso === s.href ? stili.voceAttiva : stili.voce
                }
                aria-current={percorso === s.href ? "page" : undefined}
              >
                {s.etichetta}
              </Link>
            ))}
          </nav>
        )}

        <span className={stili.chi}>
          {auth.user?.profile.email ?? auth.user?.profile.name}
        </span>
        <button
          className={stili.esci}
          onClick={() => auth.signoutRedirect()}
        >
          Esci
        </button>
      </header>

      <main className={stili.contenuto}>
        {amministratore ? (
          children
        ) : (
          <div className={stili.negato}>
            <h2>Non hai accesso alla console</h2>
            <p>
              Il tuo account non porta il ruolo <code>admin</code>. Chi
              amministra la piattaforma può assegnartelo da Keycloak.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
