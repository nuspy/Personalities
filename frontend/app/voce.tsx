"use client";

/* Ascoltare una risposta, e guardare chi la dice.
 *
 * **Il server manda audio e tempi; qui si disegna.** Un volto parlante
 * generato sul server costerebbe rendering e banda video per ogni
 * ascoltatore, e legherebbe la qualità dell'animazione alla latenza della
 * rete invece che a questa macchina.
 *
 * **Senza tempi non si anima niente.** `origine_tempi` vuoto significa che
 * nessuno li ha misurati: la bocca resta ferma e la parola non si evidenzia.
 * Muoverle su una stima produce uno sfasamento che cresce lungo la frase, e
 * l'occhio lo nota molto prima di notare una forma sbagliata.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "react-oidc-context";
import {
  ErroreApi,
  leggiVoce,
  leggiVolto,
  sorgenteAudio,
  type Voce,
  type Volto,
} from "@/lib/api";
import stili from "./voce.module.css";

/** Dove siamo nella riproduzione. */
export interface Istante {
  /** Indice della parola in corso, o -1. */
  parola: number;
  /** Forma della bocca, `X` a riposo. */
  forma: string;
  parlando: boolean;
}

const FERMO: Istante = { parola: -1, forma: "X", parlando: false };

/** Da quali origini ci si può fidare per animare.
 *
 * `stima` no: la sintesi di sviluppo deduce i tempi dalla lunghezza delle
 * parole, e l'errore si accumula lungo la frase — a metà la bocca è mezzo
 * secondo avanti. Riprodurre l'audio senza labiale è la cosa giusta da fare
 * con una stima, e distinguerla qui è l'unico posto dove si può: il server
 * l'ha già dichiarata onestamente. */
const TEMPI_ATTENDIBILI = ["allineamento", "fornitore"];

function siPuoAnimare(voce: Voce | null): voce is Voce {
  return !!voce && TEMPI_ATTENDIBILI.includes(voce.origine_tempi);
}

/** L'elemento che copre l'istante `t`, o -1.
 *
 * Ricerca binaria: i visemi di una frase lunga sono centinaia e scorrerli a
 * ogni fotogramma si sentirebbe sulle macchine lente, che sono proprio quelle
 * dove un'animazione a scatti si nota di più. */
function indiceA(
  elementi: { inizio: number; fine: number }[], t: number,
): number {
  let basso = 0;
  let alto = elementi.length - 1;
  while (basso <= alto) {
    const mezzo = (basso + alto) >> 1;
    const e = elementi[mezzo];
    if (t < e.inizio) alto = mezzo - 1;
    else if (t >= e.fine) basso = mezzo + 1;
    else return mezzo;
  }
  return -1;
}

interface Stato {
  istante: Istante;
  voce: Voce | null;
  inAttesa: boolean;
  errore: string | null;
  /** Quale turno si sta ascoltando, o null. */
  turno: number | null;
  ascolta: (turno: number, testo: string, labiale: boolean) => void;
  ferma: () => void;
}

/** La riproduzione della voce, per tutta la conversazione.
 *
 * Uno solo per pagina e non uno per turno: due risposte che suonano insieme
 * sono incomprensibili, e impedirlo dal componente del singolo turno
 * richiederebbe che ciascuno sapesse degli altri.
 */
export function useVoce(personalita: string | null): Stato {
  const auth = useAuth();
  const [voce, setVoce] = useState<Voce | null>(null);
  const [turno, setTurno] = useState<number | null>(null);
  const [inAttesa, setInAttesa] = useState(false);
  const [errore, setErrore] = useState<string | null>(null);
  const [istante, setIstante] = useState<Istante>(FERMO);
  const audio = useRef<HTMLAudioElement | null>(null);

  const libera = useCallback(() => {
    const corrente = audio.current;
    if (!corrente) return;
    corrente.pause();
    /* L'URL di un blob resta allocato finché non lo si revoca: una
     * conversazione lunga ne accumulerebbe uno per risposta ascoltata. */
    if (corrente.src.startsWith("blob:")) URL.revokeObjectURL(corrente.src);
    audio.current = null;
  }, []);

  const ferma = useCallback(() => {
    libera();
    setTurno(null);
    setVoce(null);
    setIstante(FERMO);
  }, [libera]);

  useEffect(() => ferma, [ferma]);

  const ascolta = useCallback(
    async (quale: number, testo: string, labiale: boolean) => {
      if (turno === quale) {
        ferma();
        return;
      }

      const token = auth.user?.access_token;
      if (!token) return;

      libera();
      setErrore(null);
      setInAttesa(true);
      setTurno(quale);

      try {
        const risultato = await leggiVoce(token, testo, personalita, labiale);
        const elemento = new Audio(sorgenteAudio(risultato));
        audio.current = elemento;
        setVoce(risultato);

        elemento.addEventListener("ended", () => {
          setIstante(FERMO);
          setTurno(null);
        });
        await elemento.play();
        setIstante((i) => ({ ...i, parlando: true }));
      } catch (e: unknown) {
        setErrore(
          e instanceof ErroreApi ? e.message : "Non riesco a leggere ad alta voce.",
        );
        setTurno(null);
      } finally {
        setInAttesa(false);
      }
    },
    [auth, ferma, libera, personalita, turno],
  );

  /* La posizione si ricalcola a ogni fotogramma e non su `timeupdate`:
   * quell'evento scatta quattro volte al secondo, abbastanza per evidenziare
   * una parola e troppo poco per una bocca, che a quel passo scatterebbe. */
  useEffect(() => {
    const elemento = audio.current;
    if (!elemento || !siPuoAnimare(voce)) return;

    let vivo = true;
    const passo = () => {
      if (!vivo) return;
      const t = elemento.currentTime;
      setIstante({
        parola: indiceA(voce.parole, t),
        forma: voce.visemi[indiceA(voce.visemi, t)]?.forma ?? "X",
        parlando: !elemento.paused && !elemento.ended,
      });
      requestAnimationFrame(passo);
    };
    requestAnimationFrame(passo);

    return () => {
      vivo = false;
    };
  }, [voce]);

  return { istante, voce, inAttesa, errore, turno, ascolta, ferma };
}

/* ---- il pulsante ---- */

export function BottoneAscolto({
  attivo,
  inAttesa,
  disabilitato,
  motivo,
  onClick,
}: {
  attivo: boolean;
  inAttesa: boolean;
  disabilitato: boolean;
  /** Perché è spento. Un pulsante disabilitato col motivo accanto è
   *  informazione; uno che sparisce sembra un difetto. */
  motivo?: string;
  onClick: () => void;
}) {
  return (
    <button
      className={attivo ? stili.ascoltoAttivo : stili.ascolto}
      onClick={onClick}
      disabled={disabilitato || inAttesa}
      title={disabilitato ? motivo : attivo ? "Ferma" : "Ascolta"}
      aria-label={attivo ? "Ferma la lettura" : "Ascolta la risposta"}
    >
      <span aria-hidden="true">{inAttesa ? "…" : attivo ? "◼" : "▶"}</span>
      <span className={stili.ascoltoTesto}>
        {inAttesa ? "Preparo" : attivo ? "Ferma" : "Ascolta"}
      </span>
    </button>
  );
}

/* ---- il testo che segue la voce ---- */

export function TestoParlato({
  testo,
  voce,
  parola,
}: {
  testo: string;
  voce: Voce | null;
  parola: number;
}) {
  /* Senza tempi si mostra il testo com'è: evidenziare a caso sarebbe peggio
   * che non evidenziare, perché sposta l'attenzione sulla parola sbagliata. */
  if (!siPuoAnimare(voce) || parola < 0) {
    return <>{testo}</>;
  }

  return (
    <>
      {voce.parole.map((p, i) => (
        <span key={i} className={i === parola ? stili.parolaDetta : undefined}>
          {p.testo}{" "}
        </span>
      ))}
    </>
  );
}

/* ---- il volto ---- */

export function useVolto(personalita: string | null): Volto | null {
  const [volto, setVolto] = useState<Volto | null>(null);

  useEffect(() => {
    let annullato = false;
    /* Anche il caso «nessuna personalità» passa dalla promessa: azzerare lo
     * stato nel corpo dell'effetto costringerebbe React a un secondo
     * rendering per dire una cosa già decisa. */
    const richiesta = personalita
      ? leggiVolto(personalita)
      : Promise.resolve(null);

    richiesta
      .then((v) => {
        if (!annullato) setVolto(v);
      })
      .catch(() => {
        /* Una personalità senza volto, o un volto configurato male: il
         * catalogo resta usabile col solo nome, e l'errore riguarda chi
         * amministra — non chi sta conversando. */
        if (!annullato) setVolto(null);
      });
    return () => {
      annullato = true;
    };
  }, [personalita]);

  return volto;
}

export function Ritratto({
  volto,
  istante,
}: {
  volto: Volto;
  istante: Istante;
}) {
  if (volto.tipo === "video") {
    return <RitrattoVideo volto={volto} parlando={istante.parlando} />;
  }

  if (volto.tipo === "modello") {
    /* Il rendering 3D con i morph target richiede three.js e un file da
     * provare: finché non c'è, si mostra il poster invece di un riquadro
     * vuoto. Il descrittore dice `labiale: true` e il client non lo onora —
     * ed è meglio dirlo qui che animare niente in silenzio. */
    const poster = (volto.extra?.poster as string | undefined) ?? "";
    /* `<img>` e non `next/image`: l'indirizzo è di un host arbitrario scelto
     * da chi amministra, e l'ottimizzatore pretenderebbe di elencarli tutti in
     * configurazione — un avatar aggiunto dalla console smetterebbe di
     * comparire finché qualcuno non ridispiega il frontend. */
    return poster ? (
      // eslint-disable-next-line @next/next/no-img-element
      <img className={stili.ritratto} src={poster} alt={volto.nome} />
    ) : (
      <div className={stili.ritrattoAssente} role="img" aria-label={volto.nome}>
        {volto.nome.slice(0, 1)}
      </div>
    );
  }

  return (
    <div className={stili.ritrattoContenitore}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className={stili.ritratto} src={volto.uri} alt={volto.nome} />
      {istante.parlando && volto.extra?.onda !== false && <Onda />}
    </div>
  );
}

/** Un'onda sonora sotto il ritratto fermo.
 *
 * Dice «sta parlando» senza fingere un labiale che un'immagine non può
 * avere. Tre barre e non venti: è un segnale di stato, non un analizzatore
 * di spettro, e animarne venti costerebbe più del ritratto stesso. */
function Onda() {
  return (
    <span className={stili.onda} aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}

function RitrattoVideo({
  volto,
  parlando,
}: {
  volto: Volto;
  parlando: boolean;
}) {
  const clip = (volto.extra?.clip ?? {}) as Record<string, string>;
  const sorgente = (parlando && clip.parlante) || clip.fermo || volto.uri;

  return (
    <video
      className={stili.ritratto}
      src={sorgente}
      autoPlay
      loop
      muted
      playsInline
      /* `key` sulla sorgente: senza, cambiare `src` su un elemento già in
       * riproduzione lascia il browser sul fotogramma vecchio finché non
       * carica, e il passaggio fra fermo e parlante si vede come uno scatto. */
      key={sorgente}
    />
  );
}
