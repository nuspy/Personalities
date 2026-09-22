"use client";

/* Le quattro sezioni, e l'involucro delle pagine che non sono la conversazione.
 *
 * **In alto su schermo largo, in basso sul telefono.** Sul telefono le voci
 * stanno sotto il pollice, in una barra da cinque — il massimo sensato:
 * oltre diventa un menu travestito. La pagina della conversazione non
 * ha la barra in basso: lì il posto è del campo in cui si scrive, e le
 * sezioni restano raggiungibili dalla testata.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "react-oidc-context";
import stili from "./navigazione.module.css";

export const SEZIONI = [
  { href: "/", etichetta: "Conversa", segno: "¶" },
  { href: "/conversazioni", etichetta: "Conversazioni", segno: "≡" },
  { href: "/memorie", etichetta: "Memorie", segno: "◎" },
  { href: "/crea", etichetta: "Crea", segno: "✎" },
  { href: "/piano", etichetta: "Piano", segno: "◇" },
];

/** I collegamenti in testata, per la pagina della conversazione.
 *
 * Su schermo largo sono voci in fila; sul telefono un menu che si apre con un
 * tocco, e che porta anche l'uscita: la testata della conversazione non ha
 * posto per tutto, e la barra in basso qui ruberebbe spazio al campo. Un
 * `<details>` e non uno stato: si apre e chiude da tastiera e da lettore di
 * schermo senza una riga di codice. */
export function LinkSezioni() {
  const percorso = usePathname();
  const auth = useAuth();
  return (
    <>
      <nav className={stili.inTestata} aria-label="Sezioni">
        {SEZIONI.filter((s) => s.href !== "/").map((s) => (
          <Link
            key={s.href}
            href={s.href}
            className={percorso === s.href ? stili.voceAttiva : stili.voce}
            aria-current={percorso === s.href ? "page" : undefined}
          >
            {s.etichetta}
          </Link>
        ))}
      </nav>
      <details className={stili.menu}>
        <summary className={stili.menuApri}>
          <span aria-hidden="true">☰</span>
          <span className="solo-lettori">Sezioni</span>
        </summary>
        <nav className={stili.menuVoci} aria-label="Sezioni">
          {SEZIONI.filter((s) => s.href !== "/").map((s) => (
            <Link key={s.href} href={s.href} className={stili.menuVoce}>
              <span aria-hidden="true" className={stili.segno}>{s.segno}</span>
              {s.etichetta}
            </Link>
          ))}
          <button className={stili.menuVoce} onClick={() => auth.signoutRedirect()}>
            <span aria-hidden="true" className={stili.segno}>→</span>
            Esci
          </button>
        </nav>
      </details>
    </>
  );
}

/** L'involucro delle pagine secondarie: accesso, testata, barra in basso. */
export function Involucro({
  titolo,
  sottotitolo,
  children,
}: {
  titolo: string;
  sottotitolo?: string;
  children: React.ReactNode;
}) {
  const auth = useAuth();
  const percorso = usePathname();

  if (auth.isLoading) {
    return <p className={stili.attesa}>Verifica della sessione…</p>;
  }

  if (!auth.isAuthenticated) {
    return (
      <div className={stili.soglia}>
        <h1 className={stili.titolo}>{titolo}</h1>
        <p className={stili.sottotitolo}>Serve un account per vedere questa pagina.</p>
        <button className={stili.primaria} onClick={() => auth.signinRedirect()}>
          Entra
        </button>
      </div>
    );
  }

  return (
    <div className={stili.pagina}>
      <header className={stili.testata}>
        <Link href="/" className={stili.marchio}>
          Personalities<span>.</span>
        </Link>
        <nav className={stili.navigazioneAlta} aria-label="Sezioni">
          {SEZIONI.map((s) => (
            <Link
              key={s.href}
              href={s.href}
              className={percorso === s.href ? stili.voceAttiva : stili.voce}
              aria-current={percorso === s.href ? "page" : undefined}
            >
              {s.etichetta}
            </Link>
          ))}
        </nav>
        <button className={stili.esci} onClick={() => auth.signoutRedirect()}>
          Esci
        </button>
      </header>

      <main className={stili.contenuto}>
        <h1 className={stili.titolo}>{titolo}</h1>
        {sottotitolo && <p className={stili.sottotitolo}>{sottotitolo}</p>}
        {children}
      </main>

      <nav className={stili.barraBassa} aria-label="Sezioni">
        {SEZIONI.map((s) => (
          <Link
            key={s.href}
            href={s.href}
            className={percorso === s.href ? stili.schedaAttiva : stili.scheda}
            aria-current={percorso === s.href ? "page" : undefined}
          >
            <span aria-hidden="true" className={stili.segno}>{s.segno}</span>
            {s.etichetta}
          </Link>
        ))}
      </nav>
    </div>
  );
}
