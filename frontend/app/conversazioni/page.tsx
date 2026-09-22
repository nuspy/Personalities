"use client";

/* Le conversazioni aperte: riprenderne una, o archiviarla.
 *
 * **Archiviare è anche fare posto.** Il piano limita le conversazioni aperte,
 * e il messaggio del limite rimanda qui: chiudere quelle finite è il modo per
 * aprirne di nuove senza cambiare piano. Archiviata, una conversazione passa
 * all'estrazione delle memorie come una chiusa per inattività.
 */

import Link from "next/link";
import { archiviaConversazione, elencaConversazioni, type Conversazione } from "@/lib/api";
import { useAzione, useCarica } from "@/lib/usa";
import { Involucro } from "../navigazione";
import stili from "../pagine.module.css";

function quando(iso: string | null): string {
  if (!iso) return "mai";
  const data = new Date(iso);
  const oggi = new Date();
  return data.toDateString() === oggi.toDateString()
    ? `oggi alle ${data.toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" })}`
    : data.toLocaleDateString("it-IT", { day: "numeric", month: "long", year: "numeric" });
}

function indirizzoDi(c: Conversazione): string {
  const parametri = new URLSearchParams({ conversazione: c.id });
  if (c.personality) parametri.set("personalita", c.personality);
  return `/?${parametri}`;
}

export default function PaginaConversazioni() {
  return (
    <Involucro
      titolo="Conversazioni"
      sottotitolo="Riprendi da dove eri rimasto, o archivia ciò che è finito: la personalità ne trattiene ciò che conta nelle memorie."
    >
      <Contenuto />
    </Involucro>
  );
}

function Contenuto() {
  const conversazioni = useCarica(elencaConversazioni);

  if (conversazioni.errore) return <p className={stili.errore}>{conversazioni.errore}</p>;
  if (!conversazioni.dati) return <p className={stili.nota}>Caricamento…</p>;
  if (conversazioni.dati.length === 0) {
    return (
      <div className={stili.vuoto}>
        Nessuna conversazione aperta. <Link href="/">Comincia a conversare</Link>.
      </div>
    );
  }

  return (
    <ul className={stili.elenco}>
      {conversazioni.dati.map((c) => (
        <Riga key={c.id} conversazione={c} suArchiviata={conversazioni.ricarica} />
      ))}
    </ul>
  );
}

function Riga({ conversazione: c, suArchiviata }: { conversazione: Conversazione; suArchiviata: () => void }) {
  const { esegui, inCorso, errore } = useAzione();

  return (
    <li className={stili.riga}>
      <div>
        <p className={stili.rigaTesto}>
          <Link href={indirizzoDi(c)} className={stili.titoloCollegato}>
            {c.title || "Conversazione senza titolo"}
          </Link>
        </p>
        <p className={stili.rigaDettagli}>
          {c.personality ? `${c.personality} · ` : ""}ultimo messaggio {quando(c.last_message_at)}
        </p>
      </div>
      <div className={stili.rigaAzioni}>
        <Link href={indirizzoDi(c)} className={stili.primaria} style={{ display: "inline-flex", alignItems: "center", textDecoration: "none" }}>
          Riprendi
        </Link>
        <button
          className={stili.secondaria}
          disabled={inCorso}
          onClick={async () => {
            if (await esegui((t) => archiviaConversazione(t, c.id))) suArchiviata();
          }}
        >
          Archivia
        </button>
      </div>
      {errore && <p className={stili.errore}>{errore}</p>}
    </li>
  );
}
