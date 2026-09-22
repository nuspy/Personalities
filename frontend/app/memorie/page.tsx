"use client";

/* Ciò che le personalità ricordano di te.
 *
 * **Cancellare è un diritto, non una preferenza**: ogni memoria si toglie con
 * un gesto, e «dimentica tutto» chiede conferma perché è l'unica operazione
 * che non si annulla. La storia di una memoria corretta resta visibile — alla
 * domanda «da quando lo sa?» si risponde, invece di mostrare un fatto che
 * sembra essere stato sempre così.
 */

import { useState } from "react";
import {
  accessiMemorie,
  aggiungiMemoria,
  dimenticaMemoria,
  dimenticaTutto,
  elencaMemorie,
  storiaMemoria,
  type Memoria,
} from "@/lib/api";
import { useAzione, useCarica } from "@/lib/usa";
import { Involucro } from "../navigazione";
import stili from "../pagine.module.css";

const GENERI: Record<Memoria["kind"], string> = {
  identita: "Chi sei",
  preferenza: "Preferenze",
  fatto: "Fatti",
  impegno: "Impegni",
  sessione: "Di questa sessione",
};

const ORDINE: Memoria["kind"][] = ["identita", "preferenza", "fatto", "impegno", "sessione"];

const AZIONI: Record<string, string> = {
  "memorie.lette": "ha letto le tue memorie",
  "memorie.consolidate": "ha riordinato le tue memorie",
};

function data(iso: string | null): string {
  return iso ? new Date(iso).toLocaleDateString("it-IT", { day: "numeric", month: "short", year: "numeric" }) : "—";
}

export default function PaginaMemorie() {
  return (
    <Involucro
      titolo="Le tue memorie"
      sottotitolo="Ciò che le personalità ricordano di te fra una conversazione e l'altra. Puoi aggiungere, correggere cancellando, o dimenticare tutto."
    >
      <Contenuto />
    </Involucro>
  );
}

function Contenuto() {
  const [superate, setSuperate] = useState(false);
  const memorie = useCarica((t) => elencaMemorie(t, superate), [superate]);
  const accessi = useCarica(accessiMemorie);

  return (
    <>
      <Aggiungi suFatto={memorie.ricarica} />

      <label className={stili.casella}>
        <input type="checkbox" checked={superate} onChange={(e) => setSuperate(e.target.checked)} />
        Mostra anche ciò che non vale più
      </label>

      {memorie.errore && <p className={stili.errore}>{memorie.errore}</p>}
      {memorie.dati && memorie.dati.memorie.length === 0 && (
        <div className={stili.vuoto}>
          Nessuna memoria ancora. Le personalità ricordano ciò che dici di te
          quando una conversazione si chiude; puoi anche scriverlo qui.
        </div>
      )}
      {memorie.dati &&
        ORDINE.map((genere) => {
          const del_genere = memorie.dati!.memorie.filter((m) => m.kind === genere);
          if (del_genere.length === 0) return null;
          return (
            <section key={genere} className={stili.sezione}>
              <h2 className={stili.sezioneTitolo}>
                {GENERI[genere]} · {del_genere.length}
              </h2>
              <ul className={stili.elenco}>
                {del_genere.map((m) => (
                  <RigaMemoria key={m.id} memoria={m} suCambio={memorie.ricarica} />
                ))}
              </ul>
            </section>
          );
        })}

      {memorie.dati && memorie.dati.memorie.length > 0 && (
        <DimenticaTutto suFatto={memorie.ricarica} />
      )}

      <section className={stili.sezione}>
        <h2 className={stili.sezioneTitolo}>Chi ha guardato</h2>
        {accessi.dati && accessi.dati.length === 0 && (
          <p className={stili.nota}>Nessun amministratore ha letto le tue memorie.</p>
        )}
        {accessi.dati && accessi.dati.length > 0 && (
          <ul className={stili.elenco}>
            {accessi.dati.map((a, i) => (
              <li key={i} className={stili.riga}>
                <div>
                  <p className={stili.rigaTesto}>
                    Un amministratore {AZIONI[a.action] ?? a.action} ({a.count})
                  </p>
                  <p className={stili.rigaDettagli}>
                    {new Date(a.created_at).toLocaleString("it-IT")} ·{" "}
                    {a.reason ? `motivo: ${a.reason}` : "nessun motivo indicato"}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

function RigaMemoria({ memoria, suCambio }: { memoria: Memoria; suCambio: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [storia, setStoria] = useState<Memoria[] | null>(null);

  const mostraStoria = async () => {
    if (storia) {
      setStoria(null);
      return;
    }
    const s = await esegui((t) => storiaMemoria(t, memoria.id));
    /* La catena comincia dalla memoria stessa: le versioni precedenti sono
     * il resto. */
    if (s) setStoria(s.filter((m) => m.id !== memoria.id));
  };

  return (
    <li className={stili.riga}>
      <div>
        <p className={memoria.viva ? stili.rigaTesto : `${stili.rigaTesto} ${stili.superata}`}>
          {memoria.content}
        </p>
        <p className={stili.rigaDettagli}>
          dal {data(memoria.first_seen_at)}
          {memoria.times_referenced > 0 && ` · usata ${memoria.times_referenced} volte`}
          {memoria.valid_to && ` · non vale più dal ${data(memoria.valid_to)}`}
          {memoria.expires_at && memoria.viva && ` · scade il ${data(memoria.expires_at)}`}
        </p>
      </div>
      <div className={stili.rigaAzioni}>
        <button className={stili.secondaria} onClick={mostraStoria} disabled={inCorso} aria-expanded={!!storia}>
          {storia ? "Chiudi" : "Storia"}
        </button>
        {memoria.viva && (
          <button
            className={stili.distruttiva}
            disabled={inCorso}
            onClick={async () => {
              if (await esegui((t) => dimenticaMemoria(t, memoria.id))) suCambio();
            }}
          >
            Dimentica
          </button>
        )}
      </div>
      {storia && (
        <div className={stili.storia}>
          {storia.length === 0 ? (
            <p>Nessuna versione precedente: è sempre stata così.</p>
          ) : (
            storia.map((s) => (
              <p key={s.id}>
                <span className={stili.superata}>{s.content}</span>{" "}
                <span className={stili.rigaDettagli}>
                  ({data(s.first_seen_at)} – {data(s.valid_to)})
                </span>
              </p>
            ))
          )}
        </div>
      )}
      {errore && <p className={stili.errore}>{errore}</p>}
    </li>
  );
}

function Aggiungi({ suFatto }: { suFatto: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [testo, setTesto] = useState("");
  const [genere, setGenere] = useState("fatto");

  return (
    <section className={stili.sezione}>
      <h2 className={stili.sezioneTitolo}>Aggiungi</h2>
      <form
        className={stili.modulo}
        onSubmit={async (e) => {
          e.preventDefault();
          if (testo.trim().length < 3) return;
          if (await esegui((t) => aggiungiMemoria(t, testo.trim(), genere))) {
            setTesto("");
            suFatto();
          }
        }}
      >
        <label htmlFor="memoria-nuova" className="solo-lettori">Cosa ricordare</label>
        <input
          id="memoria-nuova"
          className={stili.ingresso}
          value={testo}
          onChange={(e) => setTesto(e.target.value)}
          placeholder="Per esempio: preferisco risposte brevi"
          maxLength={2000}
        />
        <label htmlFor="memoria-genere" className="solo-lettori">Genere</label>
        <select
          id="memoria-genere"
          className={stili.ingresso}
          style={{ flex: "0 0 auto" }}
          value={genere}
          onChange={(e) => setGenere(e.target.value)}
        >
          <option value="identita">Chi sei</option>
          <option value="preferenza">Preferenza</option>
          <option value="fatto">Fatto</option>
          <option value="impegno">Impegno</option>
        </select>
        <button className={stili.primaria} disabled={inCorso || testo.trim().length < 3}>
          Ricorda
        </button>
      </form>
      {errore && <p className={stili.errore}>{errore}</p>}
    </section>
  );
}

function DimenticaTutto({ suFatto }: { suFatto: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [conferma, setConferma] = useState(false);

  return (
    <section className={stili.sezione}>
      {conferma ? (
        <div className={stili.avviso}>
          Tutte le memorie verranno cancellate, e non si torna indietro. Le
          personalità ti tratteranno come qualcuno che non conoscono.
          <div className={stili.rigaAzioni} style={{ justifyContent: "flex-start", marginTop: "0.75rem" }}>
            <button
              className={stili.distruttiva}
              disabled={inCorso}
              onClick={async () => {
                if (await esegui((t) => dimenticaTutto(t))) suFatto();
                setConferma(false);
              }}
            >
              Sì, dimentica tutto
            </button>
            <button className={stili.secondaria} onClick={() => setConferma(false)}>
              Annulla
            </button>
          </div>
        </div>
      ) : (
        <button className={stili.distruttiva} onClick={() => setConferma(true)}>
          Dimentica tutto
        </button>
      )}
      {errore && <p className={stili.errore}>{errore}</p>}
    </section>
  );
}
