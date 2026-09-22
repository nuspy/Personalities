"use client";

/* Le voci proprie: personalità e corpora che si creano da sé.
 *
 * Ciò che nasce qui è privato — lo vede solo chi l'ha creato — e segue le
 * stesse regole delle personalità del catalogo: cambiare il prompt crea una
 * versione nuova, e le conversazioni già avvenute restano legate a quella
 * che le ha prodotte. Il piano dice quante se ne possono avere.
 */

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  archiviaPersonalitaPropria,
  caricaNelCorpus,
  collegaCorpora,
  creaCorpusProprio,
  creaPersonalitaPropria,
  documentiDelCorpus,
  eliminaDocumentoProprio,
  formatiCaricabili,
  leggiCreazioni,
  modificaPersonalitaPropria,
  statoLavoro,
  type CorpusProprio,
  type Lavoro,
  type PersonalitaPropria,
} from "@/lib/api";
import { useAzione, useCarica, useToken } from "@/lib/usa";
import { Involucro } from "../navigazione";
import stili from "../pagine.module.css";
import propri from "./crea.module.css";

const MODELLO_PROMPT = `Sei [chi]: [in una riga, chi è e da dove parla].

Parli come [il suo modo: lessico, ritmo, tono]. Ti rivolgi a chi ti scrive [in che modo].

Quando non sai qualcosa, lo dici. Non inventi fatti, date o citazioni.`;

export default function PaginaCrea() {
  return (
    <Involucro
      titolo="Le tue voci"
      sottotitolo="Personalità e corpora tuoi, visibili solo a te. Scrivi chi è la voce, dalle i suoi testi, e conversaci."
    >
      <Contenuto />
    </Involucro>
  );
}

function Contenuto() {
  const creazioni = useCarica(leggiCreazioni);
  const dati = creazioni.dati;

  if (creazioni.errore) return <p className={stili.errore}>{creazioni.errore}</p>;
  if (!dati) return <p className={stili.nota}>Caricamento…</p>;

  const { permessi } = dati;
  const nessunDiritto = permessi.personalita_proprie === 0 && !permessi.corpora_propri;

  if (nessunDiritto && dati.personalita.length === 0) {
    return (
      <div className={stili.vuoto}>
        Il tuo piano non comprende voci proprie.{" "}
        <Link href="/piano">Guarda i piani che le includono</Link>.
      </div>
    );
  }

  const limite = permessi.personalita_proprie;
  const puoCreare = limite === null || permessi.personalita_usate < limite;

  return (
    <>
      <section className={stili.sezione}>
        <h2 className={stili.sezioneTitolo}>
          Personalità{limite !== null && ` · ${permessi.personalita_usate} di ${limite}`}
        </h2>
        {dati.personalita.length === 0 && (
          <div className={stili.vuoto}>Nessuna ancora: comincia da qui sotto.</div>
        )}
        <ul className={propri.schede}>
          {dati.personalita.map((p) => (
            <Scheda key={p.id} personalita={p} corpora={dati.corpora} suCambio={creazioni.ricarica} />
          ))}
        </ul>
        {puoCreare ? (
          <NuovaPersonalita suFatto={creazioni.ricarica} />
        ) : (
          <p className={stili.nota}>
            Hai raggiunto il massimo del tuo piano: archiviane una per crearne un&apos;altra,
            o <Link href="/piano">passa a un piano superiore</Link>.
          </p>
        )}
      </section>

      <section className={stili.sezione}>
        <h2 className={stili.sezioneTitolo}>Corpora</h2>
        {!permessi.corpora_propri ? (
          <p className={stili.nota}>
            Il tuo piano non comprende corpora propri: le tue voci rispondono dal prompt soltanto.
          </p>
        ) : (
          <Corpora corpora={dati.corpora} suCambio={creazioni.ricarica} />
        )}
      </section>
    </>
  );
}

/* ---- personalità ---- */

function Scheda({
  personalita: p,
  corpora,
  suCambio,
}: {
  personalita: PersonalitaPropria;
  corpora: CorpusProprio[];
  suCambio: () => void;
}) {
  const [apri, setApri] = useState<"prompt" | "corpora" | "archivia" | null>(null);
  const { esegui, inCorso, errore } = useAzione();
  const [prompt, setPrompt] = useState(p.prompt);
  const [scelti, setScelti] = useState<string[]>(p.corpora);

  const nomiCorpora = corpora.filter((c) => p.corpora.includes(c.id)).map((c) => c.nome);

  return (
    <li className={propri.scheda}>
      <div className={propri.schedaTesta}>
        <div>
          <div className={propri.schedaNome}>{p.nome}</div>
          <div className={stili.rigaDettagli}>
            versione {p.versione}
            {nomiCorpora.length > 0 ? ` · ${nomiCorpora.join(", ")}` : " · senza corpora"}
          </div>
        </div>
        <Link
          href={`/?personalita=${encodeURIComponent(p.slug)}`}
          className={stili.primaria}
          style={{ display: "inline-flex", alignItems: "center", textDecoration: "none" }}
        >
          Conversa
        </Link>
      </div>

      <div className={stili.rigaAzioni} style={{ justifyContent: "flex-start", marginTop: "0.6rem" }}>
        <button className={stili.secondaria} aria-expanded={apri === "prompt"} onClick={() => setApri(apri === "prompt" ? null : "prompt")}>
          Prompt
        </button>
        {corpora.length > 0 && (
          <button className={stili.secondaria} aria-expanded={apri === "corpora"} onClick={() => setApri(apri === "corpora" ? null : "corpora")}>
            Corpora
          </button>
        )}
        <button className={stili.distruttiva} onClick={() => setApri(apri === "archivia" ? null : "archivia")}>
          Archivia
        </button>
      </div>

      {apri === "prompt" && (
        <div className={propri.pannello}>
          <label htmlFor={`prompt-${p.id}`} className={stili.sezioneTitolo}>Chi è questa voce</label>
          <textarea
            id={`prompt-${p.id}`}
            className={propri.area}
            rows={9}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            maxLength={8000}
          />
          <p className={stili.nota}>
            Salvare crea la versione {(p.versione ?? 0) + 1}: le conversazioni già avvenute restano sulla {p.versione}.
          </p>
          <button
            className={stili.primaria}
            disabled={inCorso || prompt.trim().length < 20 || prompt === p.prompt}
            onClick={async () => {
              if (await esegui((t) => modificaPersonalitaPropria(t, p.id, { prompt }))) {
                setApri(null);
                suCambio();
              }
            }}
          >
            Salva la nuova versione
          </button>
        </div>
      )}

      {apri === "corpora" && (
        <div className={propri.pannello}>
          <fieldset className={propri.scelte}>
            <legend className={stili.sezioneTitolo}>Da quali testi risponde</legend>
            {corpora.map((c) => (
              <label key={c.id} className={stili.casella}>
                <input
                  type="checkbox"
                  checked={scelti.includes(c.id)}
                  onChange={(e) =>
                    setScelti((prima) => (e.target.checked ? [...prima, c.id] : prima.filter((x) => x !== c.id)))
                  }
                />
                {c.nome} · {c.documenti} documenti
              </label>
            ))}
          </fieldset>
          <button
            className={stili.primaria}
            disabled={inCorso}
            onClick={async () => {
              if (await esegui((t) => collegaCorpora(t, p.id, scelti))) {
                setApri(null);
                suCambio();
              }
            }}
          >
            Salva
          </button>
        </div>
      )}

      {apri === "archivia" && (
        <div className={stili.avviso} style={{ marginTop: "0.75rem" }}>
          Archiviata, non la vedi più fra le tue voci e libera un posto; le
          conversazioni già avvenute restano leggibili.
          <div className={stili.rigaAzioni} style={{ justifyContent: "flex-start", marginTop: "0.6rem" }}>
            <button
              className={stili.distruttiva}
              disabled={inCorso}
              onClick={async () => {
                if (await esegui((t) => archiviaPersonalitaPropria(t, p.id))) suCambio();
              }}
            >
              Sì, archivia
            </button>
            <button className={stili.secondaria} onClick={() => setApri(null)}>No</button>
          </div>
        </div>
      )}
      {errore && <p className={stili.errore}>{errore}</p>}
    </li>
  );
}

function NuovaPersonalita({ suFatto }: { suFatto: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [nome, setNome] = useState("");
  const [descrizione, setDescrizione] = useState("");
  const [prompt, setPrompt] = useState("");

  return (
    <form
      className={propri.nuova}
      onSubmit={async (e) => {
        e.preventDefault();
        const creata = await esegui((t) =>
          creaPersonalitaPropria(t, { nome: nome.trim(), descrizione: descrizione.trim() || undefined, prompt: prompt.trim() }),
        );
        if (creata) {
          setNome("");
          setDescrizione("");
          setPrompt("");
          suFatto();
        }
      }}
    >
      <h3 className={propri.nuovaTitolo}>Nuova personalità</h3>
      <label className={propri.campo}>
        <span className={stili.sezioneTitolo}>Nome</span>
        <input className={stili.ingresso} value={nome} onChange={(e) => setNome(e.target.value)} maxLength={120} placeholder="Ipazia di Alessandria" />
      </label>
      <label className={propri.campo}>
        <span className={stili.sezioneTitolo}>In una riga</span>
        <input className={stili.ingresso} value={descrizione} onChange={(e) => setDescrizione(e.target.value)} maxLength={500} placeholder="Matematica e filosofa, IV secolo" />
      </label>
      <label className={propri.campo}>
        <span className={stili.sezioneTitolo}>Chi è questa voce</span>
        <textarea
          className={propri.area}
          rows={8}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={MODELLO_PROMPT}
          maxLength={8000}
        />
      </label>
      <p className={stili.nota}>
        Descrivi chi parla e come, non cosa deve sapere: i fatti vengono dai
        corpora che le colleghi, e lì restano verificabili.
      </p>
      {errore && <p className={stili.errore}>{errore}</p>}
      <button className={stili.primaria} disabled={inCorso || nome.trim().length < 2 || prompt.trim().length < 20}>
        {inCorso ? "Creo…" : "Crea"}
      </button>
    </form>
  );
}

/* ---- corpora ---- */

function Corpora({ corpora, suCambio }: { corpora: CorpusProprio[]; suCambio: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [nome, setNome] = useState("");
  const [aperto, setAperto] = useState<string | null>(null);

  return (
    <>
      {corpora.length === 0 && (
        <div className={stili.vuoto}>Nessun corpus: creane uno e caricaci i testi della tua voce.</div>
      )}
      <ul className={stili.elenco}>
        {corpora.map((c) => (
          <li key={c.id} className={stili.riga}>
            <div>
              <p className={stili.rigaTesto}>{c.nome}</p>
              <p className={stili.rigaDettagli}>{c.documenti} documenti · {c.passaggi} passaggi</p>
            </div>
            <div className={stili.rigaAzioni}>
              <button className={stili.secondaria} aria-expanded={aperto === c.id} onClick={() => setAperto(aperto === c.id ? null : c.id)}>
                {aperto === c.id ? "Chiudi" : "Documenti"}
              </button>
            </div>
            {aperto === c.id && <Documenti corpus={c} suCambio={suCambio} />}
          </li>
        ))}
      </ul>
      <form
        className={stili.modulo}
        style={{ marginTop: "1rem" }}
        onSubmit={async (e) => {
          e.preventDefault();
          if (await esegui((t) => creaCorpusProprio(t, nome.trim()))) {
            setNome("");
            suCambio();
          }
        }}
      >
        <label htmlFor="corpus-nuovo" className="solo-lettori">Nome del corpus</label>
        <input id="corpus-nuovo" className={stili.ingresso} value={nome} onChange={(e) => setNome(e.target.value)} placeholder="Opere di Ipazia" maxLength={120} />
        <button className={stili.primaria} disabled={inCorso || nome.trim().length < 2}>Nuovo corpus</button>
      </form>
      {errore && <p className={stili.errore}>{errore}</p>}
    </>
  );
}

function Documenti({ corpus, suCambio }: { corpus: CorpusProprio; suCambio: () => void }) {
  const token = useToken();
  const documenti = useCarica((t) => documentiDelCorpus(t, corpus.id), [corpus.id]);
  const formati = useCarica(formatiCaricabili);
  const { esegui, inCorso, errore } = useAzione();
  const [lavoro, setLavoro] = useState<string | null>(null);
  const [stato, setStato] = useState<Lavoro | null>(null);
  const ingresso = useRef<HTMLInputElement>(null);

  /* Il caricamento si legge nel worker: qui si chiede ogni due secondi com'è
   * andata, finché non è finito. */
  useEffect(() => {
    if (!lavoro || !token) return;
    let vivo = true;
    const giro = async () => {
      try {
        const s = await statoLavoro(token, lavoro);
        if (!vivo) return;
        setStato(s);
        if (["riuscita", "fallita", "annullata"].includes(s.stato)) {
          documenti.ricarica();
          suCambio();
          return;
        }
      } catch {
        /* un giro perso non ferma il seguente */
      }
      if (vivo) setTimeout(giro, 2000);
    };
    giro();
    return () => { vivo = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lavoro, token]);

  const inMarcia = stato !== null && !["riuscita", "fallita", "annullata"].includes(stato.stato);

  return (
    <div className={propri.documenti}>
      {documenti.dati && documenti.dati.length === 0 && <p className={stili.nota}>Nessun documento ancora.</p>}
      {documenti.dati && documenti.dati.length > 0 && (
        <ul className={propri.documentiElenco}>
          {documenti.dati.map((d) => (
            <li key={d.id}>
              <span>{d.titolo}</span>
              <span className={stili.rigaDettagli}>{d.passaggi} passaggi{d.registro === "parlato" ? " · parlato" : ""}</span>
              <button
                className={stili.distruttiva}
                disabled={inCorso}
                onClick={async () => {
                  if (await esegui((t) => eliminaDocumentoProprio(t, corpus.id, d.id))) {
                    documenti.ricarica();
                    suCambio();
                  }
                }}
                aria-label={`Elimina ${d.titolo}`}
              >
                Elimina
              </button>
            </li>
          ))}
        </ul>
      )}

      <input
        ref={ingresso}
        type="file"
        multiple
        className="solo-lettori"
        tabIndex={-1}
        aria-hidden="true"
        accept={formati.dati ? Object.keys(formati.dati.formati).join(",") : undefined}
        onChange={async (e) => {
          const scelti = Array.from(e.target.files ?? []);
          e.target.value = "";
          if (!scelti.length) return;
          setStato(null);
          const risposta = await esegui((t) => caricaNelCorpus(t, corpus.id, scelti));
          if (risposta) setLavoro(risposta.build_id);
        }}
      />
      <button className={stili.primaria} disabled={inCorso || inMarcia} onClick={() => ingresso.current?.click()}>
        {inMarcia ? "Lettura in corso…" : "Carica documenti"}
      </button>
      {formati.dati && (
        <p className={stili.nota}>
          PDF, Word, EPUB, testo e altri formati; audio e video vengono trascritti. Fino a {formati.dati.file_massimi} file per volta.
        </p>
      )}

      {stato && (
        <p className={stato.stato === "fallita" ? stili.errore : stili.nota} role="status">
          {stato.stato === "riuscita" && stato.esito
            ? `Aggiunti ${stato.esito.documenti} documenti, ${stato.esito.passaggi} passaggi.` +
              (stato.esito.falliti.length ? ` Non entrati: ${stato.esito.falliti.map((f) => `${f.nome} (${f.motivo})`).join("; ")}.` : "")
            : stato.stato === "fallita"
              ? stato.errore ?? "Il caricamento non è riuscito."
              : stato.messaggio ?? "In attesa…"}
        </p>
      )}
      {errore && <p className={stili.errore}>{errore}</p>}
    </div>
  );
}

