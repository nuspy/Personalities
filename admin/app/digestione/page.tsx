"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  avviaDigestione,
  elencoBasi,
  passaggiDi,
  seguiLavoro,
  statoDigestione,
  type PassaggioEtichettato,
  type StatoDigestione,
} from "@/lib/api";
import { useAzione, useDati, useToken } from "@/lib/usa";
import comuni from "../comuni.module.css";
import stili from "./digestione.module.css";

const PER_PAGINA = 25;

/* Quante etichette mostrare accanto a un passaggio.
 *
 * Le categorie sono multi-etichetta con punteggio, e la coda è quasi sempre
 * rumore sotto 0,1: stamparla tutta riempie la scheda di numeri che non
 * distinguono un passaggio dall'altro. */
const PUNTEGGIO_MINIMO = 0.1;

export default function Digestione() {
  const { dati: basi, errore: erroreBasi } = useDati(elencoBasi);
  const [scelto, setScelto] = useState<string | null>(null);

  /* Se c'è un solo corpus è già scelto: una lista di uno è una domanda con
   * una risposta sola. Dedotto e non messo nello stato da un effetto, che
   * farebbe rendere la pagina due volte per decidere una cosa già decisa. */
  const kbId = scelto ?? (basi?.length === 1 ? basi[0].id : null);

  return (
    <>
      <div className={comuni.intestazione}>
        <div>
          <h1 className={comuni.titolo}>Digestione</h1>
          <p className={comuni.sottotitolo}>
            L&apos;ingestione taglia e vettorizza; la digestione capisce. Ogni
            passaggio riceve una categoria funzionale — a cosa serve quando si
            costruisce la personalità — e una provenienza: lo dice lui, un
            contemporaneo, o il curatore che ha stampato il volume nel 1802.
          </p>
        </div>
      </div>

      {erroreBasi && <p className={comuni.errore}>{erroreBasi}</p>}

      {basi && basi.length === 0 && (
        <div className={comuni.vuoto}>
          Nessun corpus da digerire. Creane uno in Corpora e aggiungici
          documenti.
        </div>
      )}

      {basi && basi.length > 1 && (
        <div className={stili.scelta} role="group" aria-label="Corpus">
          {basi.map((b) => (
            <button
              key={b.id}
              className={b.id === kbId ? stili.corpusAttivo : stili.corpus}
              aria-pressed={b.id === kbId}
              onClick={() => setScelto(b.id)}
            >
              {b.name}
              <span className={stili.corpusMisura}>
                {b.stats?.passaggi ?? 0} passaggi · {b.embed_model}
              </span>
            </button>
          ))}
        </div>
      )}

      {kbId && <Corpus kbId={kbId} key={kbId} />}
    </>
  );
}

/* ---- un corpus: stato, avvio, passaggi ---- */

function Corpus({ kbId }: { kbId: string }) {
  const { dati: stato, errore, inCorso, ricarica } = useDati<StatoDigestione>(
    useCallback((t: string) => statoDigestione(t, kbId), [kbId]),
  );

  if (errore) return <p className={comuni.errore}>{errore}</p>;
  if (inCorso && !stato) return <p className={comuni.caricamento}>Caricamento…</p>;
  if (!stato) return null;

  const { passaggi: n } = stato;

  return (
    <>
      <div className={stili.cruscotto}>
        <Misura valore={n.totale} etichetta="passaggi" />
        <Misura valore={n.etichettati} etichetta="etichettati" />
        <Misura valore={n.da_fare} etichetta="da fare" />
        <Misura valore={n.scartati} etichetta="scartati" allarme />
        <Misura valore={n.ripuliti} etichetta="ripuliti" />
      </div>

      {n.totale === 0 ? (
        <div className={comuni.vuoto}>
          Il corpus non ha passaggi: prima vanno ingeriti i documenti.
        </div>
      ) : (
        <>
          <Avvio kbId={kbId} daFare={n.da_fare} alTermine={ricarica} />

          {n.etichettati > 0 && (
            <div className={stili.distribuzioni}>
              <section>
                <h2 className={comuni.riquadroTitolo}>Per categoria</h2>
                <Barre valori={stato.per_categoria} />
              </section>
              <section>
                <h2 className={comuni.riquadroTitolo}>Per provenienza</h2>
                <Barre valori={stato.per_provenienza} />
              </section>
            </div>
          )}

          <Passaggi kbId={kbId} categorie={Object.keys(stato.per_categoria)} />
        </>
      )}
    </>
  );
}

function Misura({
  valore,
  etichetta,
  allarme,
}: {
  valore: number;
  etichetta: string;
  allarme?: boolean;
}) {
  return (
    <div className={allarme && valore > 0 ? `${stili.misura} ${stili.misuraAllarme}` : stili.misura}>
      <span className={stili.misuraValore}>{valore}</span>
      <span className={stili.misuraEtichetta}>{etichetta}</span>
    </div>
  );
}

function Barre({ valori }: { valori: Record<string, number> }) {
  const voci = Object.entries(valori);
  if (voci.length === 0) return <p className={comuni.vuoto}>Nulla di etichettato.</p>;

  const massimo = Math.max(...voci.map(([, q]) => q));

  return (
    <ul className={stili.barre}>
      {voci.map(([nome, quanti]) => (
        <li
          key={nome}
          className={nome === "apparato" ? `${stili.barra} ${stili.barraScarto}` : stili.barra}
        >
          <span className={stili.barraNome}>{nome}</span>
          <span className={stili.barraTraccia}>
            <span
              className={stili.barraRiempimento}
              style={{ width: `${(quanti / massimo) * 100}%` }}
            />
          </span>
          <span className={stili.barraQuanti}>{quanti}</span>
        </li>
      ))}
    </ul>
  );
}

/* ---- avvio del lavoro ---- */

interface Riga {
  id: number;
  progress: number;
  message: string;
  level: string;
}

function Avvio({
  kbId,
  daFare,
  alTermine,
}: {
  kbId: string;
  daFare: number;
  alTermine: () => void;
}) {
  const token = useToken();
  const { esegui, inCorso, errore } = useAzione();
  const [chi, setChi] = useState("");
  const [rifai, setRifai] = useState(false);
  const [lavoro, setLavoro] = useState<string | null>(null);
  const [righe, setRighe] = useState<Riga[]>([]);
  const [finito, setFinito] = useState<string | null>(null);
  const coda = useRef<HTMLUListElement>(null);

  /* Il flusso si segue finché la pagina è aperta, e si chiude quando non lo è
   * più: una digestione dura ore, e una connessione lasciata appesa a un job
   * che nessuno guarda resta aperta fino al timeout del proxy. */
  useEffect(() => {
    if (!lavoro || !token) return;

    const smetti = seguiLavoro(token, lavoro, (evento, dato) => {
      if (evento === "progress") {
        setRighe((prima) => [
          ...prima.slice(-199),
          {
            id: Number(dato.id),
            progress: Number(dato.progress ?? 0),
            message: String(dato.message ?? ""),
            level: String(dato.level ?? "info"),
          },
        ]);
      } else if (evento === "done") {
        setFinito(String(dato.status ?? "conclusa"));
        alTermine();
      } else if (evento === "errore" || evento === "timeout") {
        setFinito(String(dato.message ?? "flusso interrotto"));
      }
    });

    return smetti;
  }, [lavoro, token, alTermine]);

  useEffect(() => {
    coda.current?.scrollTo({ top: coda.current.scrollHeight });
  }, [righe]);

  const ultima = righe[righe.length - 1];
  const inMarcia = lavoro !== null && finito === null;

  return (
    <section className={comuni.riquadro}>
      <h2 className={comuni.riquadroTitolo}>Avvia la digestione</h2>
      <p className={comuni.riquadroNota}>
        Sono ore su un corpus vero, e si accoda come una realizzazione: chiudere
        questa pagina non la ferma. Il lavoro non ha bisogno di acceleratore —
        interroga un modello, non ne addestra uno — quindi basta un worker
        qualunque.
      </p>

      <div className={stili.avvio}>
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Di chi parla il corpus</span>
          <input
            className={comuni.ingresso}
            value={chi}
            onChange={(e) => setChi(e.target.value)}
            placeholder="Lucio Anneo Seneca"
            disabled={inMarcia}
          />
          <span className={comuni.campoAiuto}>
            Cambia il giudizio sulla provenienza: senza un nome, «lo dice lui» e
            «lo dice un posteriore» si distinguono solo dal tono, e il tono
            inganna.
          </span>
        </label>

        <label className={stili.spunta}>
          <input
            type="checkbox"
            checked={rifai}
            onChange={(e) => setRifai(e.target.checked)}
            disabled={inMarcia}
          />
          Rianalizza anche i già etichettati
        </label>

        <button
          className={comuni.primaria}
          disabled={inCorso || inMarcia || (daFare === 0 && !rifai)}
          onClick={async () => {
            setRighe([]);
            setFinito(null);
            const esito = await esegui((t) => avviaDigestione(t, kbId, { chi, rifai }));
            if (esito) setLavoro(esito.build_id);
          }}
        >
          {inMarcia ? "In corso…" : "Digerisci"}
        </button>
      </div>

      {daFare === 0 && !rifai && !lavoro && (
        <p className={comuni.riquadroNota}>
          Tutti i passaggi sono già etichettati. Spunta «rianalizza» se il
          classificatore o la tassonomia sono cambiati: lasciare il corpus metà
          vecchio e metà nuovo lo rende incoerente.
        </p>
      )}

      {errore && <p className={comuni.errore}>{errore}</p>}

      {lavoro && (
        <div className={stili.avanzamento}>
          <div
            className={stili.avanzamentoTraccia}
            role="progressbar"
            aria-valuenow={ultima?.progress ?? 0}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Avanzamento della digestione"
          >
            <div
              className={stili.avanzamentoRiempimento}
              style={{ width: `${ultima?.progress ?? 0}%` }}
            />
          </div>
          <p className={stili.avanzamentoRiga}>
            <span>{finito ?? ultima?.message ?? "In attesa di un worker…"}</span>
            <span>{ultima?.progress ?? 0}%</span>
          </p>

          {righe.length > 0 && (
            <ul className={stili.diario} ref={coda}>
              {righe.map((r) => (
                <li
                  key={r.id}
                  className={r.level === "errore" ? stili.diarioErrore : undefined}
                >
                  <span className={stili.diarioPercento}>{r.progress}%</span>
                  <span>{r.message}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

/* ---- i passaggi con le loro etichette ---- */

function Passaggi({ kbId, categorie }: { kbId: string; categorie: string[] }) {
  const [categoria, setCategoria] = useState<string | null>(null);
  const [soloScartati, setSoloScartati] = useState(false);
  const [soloRipuliti, setSoloRipuliti] = useState(false);
  const [quanti, setQuanti] = useState(PER_PAGINA);

  const { dati, errore, inCorso } = useDati(
    useCallback(
      (t: string) =>
        passaggiDi(t, kbId, {
          categoria: categoria ?? undefined,
          solo_scartati: soloScartati,
          solo_ripuliti: soloRipuliti,
          limite: quanti,
        }),
      [kbId, categoria, soloScartati, soloRipuliti, quanti],
    ),
  );

  const cambia = (azione: () => void) => {
    azione();
    setQuanti(PER_PAGINA);
  };

  return (
    <>
      <div className={stili.filtri}>
        <button
          className={categoria === null ? stili.filtroAttivo : stili.filtro}
          onClick={() => cambia(() => setCategoria(null))}
        >
          tutte
        </button>
        {categorie
          .filter((c) => c !== "(nessuna)")
          .map((c) => (
            <button
              key={c}
              className={categoria === c ? stili.filtroAttivo : stili.filtro}
              onClick={() => cambia(() => setCategoria(c))}
            >
              {c}
            </button>
          ))}

        <button
          className={soloScartati ? stili.filtroScartoAttivo : stili.filtroScarto}
          onClick={() => cambia(() => setSoloScartati((v) => !v))}
        >
          scartati
        </button>
        <button
          className={soloRipuliti ? stili.filtroAttivo : stili.filtro}
          onClick={() => cambia(() => setSoloRipuliti((v) => !v))}
        >
          ripuliti
        </button>

        {dati && <span className={stili.conteggio}>{dati.totale} passaggi</span>}
      </div>

      {errore && <p className={comuni.errore}>{errore}</p>}
      {inCorso && !dati && <p className={comuni.caricamento}>Caricamento…</p>}

      {dati && dati.passaggi.length === 0 && (
        <div className={comuni.vuoto}>
          {soloRipuliti
            ? "Nessun passaggio è stato ripulito: la digestione non ha tolto prefissi da nessuno."
            : soloScartati
              ? "Nulla è stato scartato."
              : "Nessun passaggio con questo filtro."}
        </div>
      )}

      {dati && dati.passaggi.length > 0 && (
        <>
          <div className={stili.passaggi}>
            {dati.passaggi.map((p) => (
              <Passaggio key={p.id} p={p} />
            ))}
          </div>

          {dati.passaggi.length < dati.totale && (
            <button
              className={`${comuni.secondaria} ${stili.ancora}`}
              disabled={inCorso}
              onClick={() => setQuanti((n) => Math.min(n + PER_PAGINA, 200))}
            >
              {inCorso ? "…" : `Mostra altri ${PER_PAGINA}`}
            </button>
          )}
        </>
      )}
    </>
  );
}

function Passaggio({ p }: { p: PassaggioEtichettato }) {
  const punteggi = Object.entries(p.categorie)
    .filter(([, v]) => v >= PUNTEGGIO_MINIMO)
    .sort(([, a], [, b]) => b - a);

  return (
    <article className={p.scartato ? stili.passaggioScartato : stili.passaggio}>
      <header className={stili.passaggioTestata}>
        <span className={stili.fonte}>
          #{p.ordinale} · {p.documento}
          {p.sezione ? ` · ${p.sezione}` : ""}
        </span>
        {p.scartato && <span className={stili.segnoScarto}>scartato</span>}
        {p.categoria && <span className={stili.segnoCategoria}>{p.categoria}</span>}
        {p.provenienza && <span className={stili.segno}>{p.provenienza}</span>}
        {p.qualita !== null && (
          <span className={stili.segno} title="Qualità del testo">
            q {p.qualita.toFixed(2)}
          </span>
        )}
        {p.categoria === null && !p.scartato && (
          <span className={stili.segno}>da digerire</span>
        )}
      </header>

      {p.sintesi && <p className={stili.sintesi}>{p.sintesi}</p>}

      <Testo testo={p.testo} originale={p.testo_originale} />

      {p.motivo_scarto && <p className={stili.motivo}>{p.motivo_scarto}</p>}

      {punteggi.length > 0 && (
        <div className={stili.punteggi}>
          {punteggi.map(([nome, valore]) => (
            <span key={nome} className={stili.punteggio}>
              {nome} {valore.toFixed(2)}
            </span>
          ))}
        </div>
      )}
    </article>
  );
}

/** Il testo, e — quando c'è stata una pulizia — cosa è cambiato.
 *
 * Mostrare la differenza invece del solo risultato è l'unico modo di
 * accorgersi che la pulizia abbia portato via prosa dell'autore insieme
 * all'apparato del curatore. Col solo testo finale l'errore è invisibile:
 * resta un passaggio plausibile, semplicemente più corto.
 *
 * Parola per parola e non «prefisso tolto», perché le due passate si
 * sovrappongono: la riparazione del capolettera salda «C onsilio» in
 * «Consilio» nello stesso passaggio in cui il rimando del curatore viene
 * tolto dalla testa, e la differenza smette di essere un prefisso. Con il
 * solo confronto di prefisso quel caso — il più frequente — finiva in un
 * ripiego che mostrava i due testi interi e lasciava il confronto a chi
 * guardava.
 */
function Testo({ testo, originale }: { testo: string; originale: string | null }) {
  if (!originale || originale === testo) {
    return <p className={stili.testo}>{testo}</p>;
  }

  return (
    <p className={stili.testo}>
      {differenza(originale, testo).map((pezzo, i) =>
        pezzo.come === "uguale" ? (
          <span key={i}>{pezzo.testo}</span>
        ) : (
          <span
            key={i}
            className={pezzo.come === "tolto" ? stili.tolto : stili.messo}
            title={pezzo.come === "tolto" ? "tolto dalla digestione" : "aggiunto"}
          >
            {pezzo.testo}
          </span>
        ),
      )}
    </p>
  );
}

interface Pezzo {
  come: "uguale" | "tolto" | "messo";
  testo: string;
}

/** Differenza parola per parola fra due testi.
 *
 * Sottosequenza comune più lunga sulle parole: i passaggi sono poche decine
 * di parole e la tabella quadratica non si sente, mentre un confronto
 * carattere per carattere spezzerebbe le parole a metà rendendo illeggibile
 * proprio ciò che si vuole leggere.
 */
function differenza(prima: string, dopo: string): Pezzo[] {
  const a = prima.split(/(\s+)/);
  const b = dopo.split(/(\s+)/);

  const lung: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array(b.length + 1).fill(0),
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lung[i][j] =
        a[i] === b[j]
          ? lung[i + 1][j + 1] + 1
          : Math.max(lung[i + 1][j], lung[i][j + 1]);
    }
  }

  const pezzi: Pezzo[] = [];
  const aggiungi = (come: Pezzo["come"], testo: string) => {
    const ultimo = pezzi[pezzi.length - 1];
    if (ultimo && ultimo.come === come) ultimo.testo += testo;
    else pezzi.push({ come, testo });
  };

  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      aggiungi("uguale", a[i]);
      i++;
      j++;
    } else if (lung[i + 1][j] >= lung[i][j + 1]) {
      aggiungi("tolto", a[i++]);
    } else {
      aggiungi("messo", b[j++]);
    }
  }
  while (i < a.length) aggiungi("tolto", a[i++]);
  while (j < b.length) aggiungi("messo", b[j++]);

  return pezzi;
}
