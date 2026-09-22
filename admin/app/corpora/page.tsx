"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import {
  caricaDocumenti,
  creaBase,
  documentiDi,
  elencoBasi,
  eliminaDocumento,
  formatiCaricabili,
  type Base,
  type Documento,
  type EsitoIngestione,
  type FormatiCaricabili,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import { Avanzamento } from "../avanzamento";
import stili from "../comuni.module.css";
import propri from "./corpora.module.css";

/* Le lingue che l'indice testuale sa trattare, con quella che la base usa
 * già in cima: cambiarla a metà renderebbe i nuovi passaggi invisibili alla
 * ricerca per parole. */
const LINGUE: Record<string, string> = {
  it: "italiano",
  la: "latino",
  en: "inglese",
  fr: "francese",
  de: "tedesco",
  es: "spagnolo",
  pt: "portoghese",
  nl: "olandese",
  ru: "russo",
};

const CONFIGURAZIONI: Record<string, string> = {
  italian: "it", english: "en", french: "fr", german: "de", spanish: "es",
  portuguese: "pt", dutch: "nl", russian: "ru",
};

function misura(byte: number): string {
  if (byte < 1024) return `${byte} B`;
  if (byte < 1024 * 1024) return `${(byte / 1024).toFixed(0)} KB`;
  return `${(byte / (1024 * 1024)).toFixed(1)} MB`;
}

function estensione(nome: string): string {
  const punto = nome.lastIndexOf(".");
  return punto >= 0 ? nome.slice(punto).toLowerCase() : "";
}

/** Da dove viene un documento, detto in poche parole. */
function origine(d: Documento): React.ReactNode {
  if (d.uri && /^https?:\/\//.test(d.uri)) {
    return (
      <a href={d.uri} target="_blank" rel="noopener noreferrer" className={stili.collegamento}>
        {new URL(d.uri).hostname}
      </a>
    );
  }
  /* Un file caricato o un percorso locale: `new URL` su questi solleverebbe
   * un'eccezione, e l'intera tabella sparirebbe per un documento. */
  if (d.meta?.origine === "console") return `caricato · ${String(d.meta.file ?? "")}`;
  return d.uri ?? "—";
}

export default function Corpora() {
  const { dati, errore, inCorso, ricarica } = useDati(elencoBasi);
  const [aperta, setAperta] = useState(false);
  const [espansa, setEspansa] = useState<string | null>(null);

  return (
    <>
      <div className={stili.intestazione}>
        <div>
          <h1 className={stili.titolo}>Corpora</h1>
          <p className={stili.sottotitolo}>
            Ogni base registra con quale modello è stata vettorizzata e in che
            lingua è indicizzata. Sono i due valori che non possono cambiare
            dopo: interrogare con un modello diverso da quello che ha scritto i
            vettori dà risultati privi di senso, senza dare errore.
          </p>
        </div>
        <button className={stili.primaria} onClick={() => setAperta((v) => !v)}>
          {aperta ? "Annulla" : "Nuova base"}
        </button>
      </div>

      {aperta && (
        <ModuloNuova
          alTermine={() => {
            setAperta(false);
            ricarica();
          }}
        />
      )}

      {errore && <p className={stili.errore}>{errore}</p>}
      {inCorso && <p className={stili.caricamento}>Caricamento…</p>}

      {dati && dati.length === 0 && (
        <div className={stili.vuoto}>
          Nessun corpus. Creane uno, poi aggiungici documenti.
        </div>
      )}

      {dati && dati.length > 0 && (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr>
                <th>Nome</th>
                <th>Tipo</th>
                <th>Modello</th>
                <th>Lingua</th>
                <th>Documenti</th>
                <th>Passaggi</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {dati.map((b) => (
                <tr key={b.id}>
                  <td>
                    <strong>{b.name}</strong>
                    <div className={stili.mono}>{b.slug}</div>
                  </td>
                  <td className={stili.mono}>{b.kind}</td>
                  <td className={stili.mono}>{b.embed_model}</td>
                  <td className={stili.mono}>{b.text_config}</td>
                  <td className={stili.numero}>{b.stats?.documenti ?? 0}</td>
                  <td className={stili.numero}>{b.stats?.passaggi ?? 0}</td>
                  <td>
                    <button
                      className={stili.secondaria}
                      onClick={() =>
                        setEspansa(espansa === b.id ? null : b.id)
                      }
                    >
                      {espansa === b.id ? "Chiudi" : "Documenti"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {espansa && dati && (
        <Documenti
          base={dati.find((b) => b.id === espansa)!}
          allaModifica={ricarica}
        />
      )}
    </>
  );
}

function Documenti({
  base,
  allaModifica,
}: {
  base: Base;
  allaModifica: () => void;
}) {
  const { dati, errore, inCorso, ricarica } = useDati(
    (t) => documentiDi(t, base.id),
    [base.id],
  );
  const { esegui, inCorso: inAzione } = useAzione();
  const [daEliminare, setDaEliminare] = useState<string | null>(null);

  return (
    <div className={stili.riquadro} style={{ marginTop: "1.25rem" }}>
      <h2 className={stili.riquadroTitolo}>Documenti di «{base.name}»</h2>
      <p className={stili.riquadroNota}>
        Eliminare un documento porta via anche i suoi passaggi. È cancellazione
        vera e non archiviazione: un testo raccolto per errore continuerebbe
        altrimenti a comparire fra le fonti di ogni risposta.
      </p>

      {errore && <p className={stili.errore}>{errore}</p>}
      {inCorso && <p className={stili.caricamento}>Caricamento…</p>}

      <Caricamento
        base={base}
        suFatto={() => {
          ricarica();
          allaModifica();
        }}
      />

      {dati && dati.length === 0 && (
        <div className={stili.vuoto}>
          Nessun documento ancora: caricane qui sopra.
        </div>
      )}

      {dati && dati.length > 0 && (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr>
                <th>Titolo</th>
                <th>Registro</th>
                <th>Origine</th>
                <th>Impronta</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {dati.map((d) => (
                <tr key={d.id}>
                  <td>{d.title}</td>
                  <td className={stili.mono}>{String(d.meta?.registro ?? "—")}</td>
                  <td className={stili.mono}>{origine(d)}</td>
                  <td className={stili.mono}>{d.sha256}</td>
                  <td>
                    {daEliminare === d.id ? (
                      <span className={stili.riga}>
                        <button
                          className={stili.distruttiva}
                          disabled={inAzione}
                          onClick={async () => {
                            await esegui((t) =>
                              eliminaDocumento(t, base.id, d.id),
                            );
                            setDaEliminare(null);
                            ricarica();
                            allaModifica();
                          }}
                        >
                          Confermo
                        </button>
                        <button
                          className={stili.secondaria}
                          onClick={() => setDaEliminare(null)}
                        >
                          No
                        </button>
                      </span>
                    ) : (
                      <button
                        className={stili.distruttiva}
                        onClick={() => setDaEliminare(d.id)}
                      >
                        Elimina
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ---- caricamento ---- */

function Caricamento({ base, suFatto }: { base: Base; suFatto: () => void }) {
  const { dati: formati } = useDati(formatiCaricabili);
  const { esegui, inCorso, errore } = useAzione();
  const [scelti, setScelti] = useState<File[]>([]);
  const [lingua, setLingua] = useState(CONFIGURAZIONI[base.text_config] ?? "it");
  const [sopra, setSopra] = useState(false);
  const [lavoro, setLavoro] = useState<string | null>(null);
  const [finito, setFinito] = useState(false);
  const [esito, setEsito] = useState<EsitoIngestione | null>(null);
  const ingresso = useRef<HTMLInputElement>(null);

  const problema = (f: File, tutti: FormatiCaricabili | null): string | null => {
    if (!tutti) return null;
    if (!tutti.formati[estensione(f.name)]) return "formato non supportato";
    if (f.size > tutti.byte_massimi) return `oltre ${misura(tutti.byte_massimi)}`;
    return null;
  };

  const aggiungi = (lista: FileList | null) => {
    if (!lista) return;
    setScelti((prima) => {
      /* Lo stesso file scelto due volte resta uno: il servizio lo
       * salterebbe comunque, ma lo mostrerebbe come «già presente». */
      const chiave = (f: File) => `${f.name}:${f.size}`;
      const presenti = new Set(prima.map(chiave));
      return [...prima, ...Array.from(lista).filter((f) => !presenti.has(chiave(f)))];
    });
  };

  const inMarcia = lavoro !== null && !finito;
  const validi = scelti.filter((f) => !problema(f, formati));
  const troppi = formati ? scelti.length > formati.file_massimi : false;
  const conProblemi = scelti.length - validi.length;

  return (
    <div style={{ marginBottom: "1.25rem" }}>
      <div
        className={sopra ? propri.zonaAttiva : propri.zona}
        onDragOver={(e) => {
          e.preventDefault();
          setSopra(true);
        }}
        onDragLeave={() => setSopra(false)}
        onDrop={(e) => {
          e.preventDefault();
          setSopra(false);
          if (!inMarcia) aggiungi(e.dataTransfer.files);
        }}
      >
        <p className={propri.zonaTesto}>
          Trascina qui i documenti, o sceglili. Si leggono nel worker: questa
          pagina si può chiudere, il lavoro continua. Audio e video vengono
          trascritti e marcati come parlato.
        </p>
        <input
          ref={ingresso}
          type="file"
          multiple
          className="solo-lettori"
          tabIndex={-1}
          aria-hidden="true"
          accept={formati ? Object.keys(formati.formati).join(",") : undefined}
          onChange={(e) => {
            aggiungi(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          className={stili.secondaria}
          disabled={inMarcia}
          onClick={() => ingresso.current?.click()}
        >
          Scegli i file
        </button>
        {formati && (
          <span className={propri.formati}>
            {Object.keys(formati.formati).join(" ")} · fino a {misura(formati.byte_massimi)} ciascuno,{" "}
            {formati.file_massimi} per volta
          </span>
        )}
      </div>

      {scelti.length > 0 && (
        <ul className={propri.scelti}>
          {scelti.map((f) => {
            const p = problema(f, formati);
            return (
              <li key={`${f.name}:${f.size}`} className={propri.scelto}>
                <span className={propri.sceltoNome}>{f.name}</span>
                <span className={propri.sceltoMisura}>{misura(f.size)}</span>
                <button
                  type="button"
                  className={stili.secondaria}
                  disabled={inMarcia}
                  onClick={() => setScelti((prima) => prima.filter((x) => x !== f))}
                  aria-label={`Togli ${f.name}`}
                >
                  Togli
                </button>
                {p && <span className={propri.sceltoProblema}>{p}</span>}
              </li>
            );
          })}
        </ul>
      )}

      {scelti.length > 0 && (
        <div className={propri.comandi}>
          <label className={stili.campo}>
            <span className={stili.campoEtichetta}>Lingua dei documenti</span>
            <select
              className={stili.selezione}
              value={lingua}
              onChange={(e) => setLingua(e.target.value)}
              disabled={inMarcia}
            >
              {Object.entries(LINGUE).map(([codice, nome]) => (
                <option key={codice} value={codice}>{nome}</option>
              ))}
            </select>
          </label>
          <button
            className={stili.primaria}
            disabled={inCorso || inMarcia || validi.length === 0 || conProblemi > 0 || troppi}
            onClick={async () => {
              const risposta = await esegui((t) => caricaDocumenti(t, base.id, validi, lingua));
              if (risposta) {
                setEsito(null);
                setFinito(false);
                setLavoro(risposta.build_id);
                setScelti([]);
              }
            }}
          >
            {inCorso ? "Invio…" : `Carica ${validi.length} file`}
          </button>
          {conProblemi > 0 && (
            <span className={propri.esitoProblema}>
              Togli i file segnati: non entrerebbero.
            </span>
          )}
          {troppi && formati && (
            <span className={propri.esitoProblema}>
              Al massimo {formati.file_massimi} per volta.
            </span>
          )}
        </div>
      )}

      {errore && <p className={stili.errore}>{errore}</p>}

      {lavoro && (
        <Avanzamento
          key={lavoro}
          lavoro={lavoro}
          etichetta="Avanzamento del caricamento"
          suFine={(build) => {
            setFinito(true);
            const meta = build.artifact_meta as EsitoIngestione | null | undefined;
            if (meta) setEsito(meta);
            if (!build.interrotto) suFatto();
          }}
        />
      )}

      {esito && (
        <div className={propri.esito} role="status">
          <strong>
            {esito.documenti} {esito.documenti === 1 ? "documento aggiunto" : "documenti aggiunti"},{" "}
            {esito.passaggi} passaggi.
          </strong>
          {esito.documenti > 0 && (
            <>
              {" "}I passaggi nuovi non sono ancora classificati:{" "}
              <Link href="/digestione" className={stili.collegamento}>digeriscili</Link>{" "}
              prima di usarli per l&apos;addestramento.
            </>
          )}
          {(esito.saltati.length > 0 || esito.falliti.length > 0 || esito.avvisi.length > 0) && (
            <ul>
              {esito.saltati.map((x) => (
                <li key={`s-${x.nome}`}>{x.nome}: {x.motivo}</li>
              ))}
              {esito.falliti.map((x) => (
                <li key={`f-${x.nome}`} className={propri.esitoProblema}>{x.nome}: {x.motivo}</li>
              ))}
              {esito.avvisi.map((x, i) => (
                <li key={`a-${i}`} className={propri.esitoAvviso}>{x.nome}: {x.avviso}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function ModuloNuova({ alTermine }: { alTermine: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [slug, setSlug] = useState("");
  const [nome, setNome] = useState("");
  const [tipo, setTipo] = useState("corpus");

  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>Nuova base</h2>
      <p className={stili.riquadroNota}>
        Il modello di vettorizzazione non si sceglie: è quello configurato
        sulla piattaforma, e viene registrato sulla base perché le domande di
        domani siano confrontabili con i vettori di oggi.
      </p>

      <div className={stili.riga}>
        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="kbnome">
            Nome
          </label>
          <input
            id="kbnome"
            className={stili.ingresso}
            value={nome}
            onChange={(e) => setNome(e.target.value)}
            placeholder="Lettere a Lucilio"
          />
        </div>
        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="kbslug">
            Slug
          </label>
          <input
            id="kbslug"
            className={stili.ingresso}
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
            placeholder="seneca-lettere"
          />
        </div>
        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="kbtipo">
            Tipo
          </label>
          <select
            id="kbtipo"
            className={stili.selezione}
            value={tipo}
            onChange={(e) => setTipo(e.target.value)}
          >
            <option value="corpus">corpus — gli scritti</option>
            <option value="reference">riferimento — contesto</option>
            <option value="news">notizie — cambia nel tempo</option>
          </select>
        </div>
      </div>

      {errore && <p className={stili.errore}>{errore}</p>}

      <button
        className={stili.primaria}
        disabled={inCorso || !nome.trim() || !slug.trim()}
        onClick={async () => {
          const fatta = await esegui((t) =>
            creaBase(t, { slug, name: nome.trim(), kind: tipo }),
          );
          if (fatta) alTermine();
        }}
      >
        {inCorso ? "Creo…" : "Crea"}
      </button>
    </div>
  );
}
