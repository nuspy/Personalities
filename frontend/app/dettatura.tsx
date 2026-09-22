"use client";

/* Il pulsante per dettare.
 *
 * **Due strade, un solo esito: il testo finisce nella bozza, mai spedito.**
 * Il riconoscimento del browser è la prima — immediato, e l'audio non passa
 * dal nostro servizio. Dove manca (Firefox, alcune WebView) o si rifiuta
 * (Chrome senza rete verso il suo servizio), si registra e si trascrive sul
 * server. In entrambi i casi chi parla rilegge prima di mandare: una frase
 * capita male e spedita da sola produce una risposta perfetta a una domanda
 * che nessuno ha fatto, ed è peggio di nessuna risposta.
 */

import { useEffect, useRef, useState } from "react";
import { ErroreApi, trascriviRegistrazione } from "@/lib/api";
import { useToken } from "@/lib/usa";
import stili from "./dettatura.module.css";

/* Il riconoscimento del browser non ha ancora i tipi nella libreria DOM di
 * TypeScript: basta la parte che si usa. */
interface Riconoscimento {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((e: EventoRisultato) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

interface EventoRisultato {
  resultIndex: number;
  results: ArrayLike<{ isFinal: boolean; 0: { transcript: string; confidence: number } }>;
}

type CostruttoreRiconoscimento = new () => Riconoscimento;

function riconoscimentoDelBrowser(): CostruttoreRiconoscimento | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: CostruttoreRiconoscimento;
    webkitSpeechRecognition?: CostruttoreRiconoscimento;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

function puoRegistrare(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof MediaRecorder !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia
  );
}

/** Il formato che questo browser sa registrare: Safari non fa webm. */
function formatoDiRegistrazione(): string {
  for (const tipo of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]) {
    if (MediaRecorder.isTypeSupported(tipo)) return tipo;
  }
  return "";
}

/* Sotto questa fiducia il testo si fa rileggere con un avviso. Chrome
 * restituisce 0 quando non la calcola: zero vuol dire «non so», non «male». */
const FIDUCIA_MINIMA = 0.6;

/* Il server accetta un paio di minuti: oltre, conviene caricare un documento. */
const SECONDI_MASSIMI = 110;

/* Errori del riconoscimento del browser per cui ha senso passare al server:
 * non dipendono da chi parla, ma dal servizio del browser. */
const ERRORI_DEL_SERVIZIO = new Set(["network", "service-not-allowed", "language-not-supported"]);

type Stato = "fermo" | "ascolta" | "registra" | "trascrive";

export function Dettatura({
  suTesto,
  suAvviso,
  disabilitato = false,
}: {
  suTesto: (testo: string) => void;
  suAvviso: (avviso: string | null) => void;
  disabilitato?: boolean;
}) {
  const token = useToken();
  const [stato, setStato] = useState<Stato>("fermo");
  /* Diventa vero quando il riconoscimento del browser si è rifiutato per
   * ragioni sue: da lì in poi si registra, senza riprovare ogni volta. */
  const [soloServer, setSoloServer] = useState(false);

  const riconoscimento = useRef<Riconoscimento | null>(null);
  const registratore = useRef<MediaRecorder | null>(null);
  const scadenza = useRef<ReturnType<typeof setTimeout> | null>(null);

  const Browser = soloServer ? null : riconoscimentoDelBrowser();
  const disponibile = !!Browser || puoRegistrare();

  useEffect(
    () => () => {
      riconoscimento.current?.abort();
      if (registratore.current?.state === "recording") registratore.current.stop();
      if (scadenza.current) clearTimeout(scadenza.current);
    },
    [],
  );

  const ascoltaNelBrowser = (Costruttore: CostruttoreRiconoscimento) => {
    const r = new Costruttore();
    r.lang = "it-IT";
    r.interimResults = true;
    r.continuous = true;
    let incerto = false;

    r.onresult = (e) => {
      let provvisorio = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const risultato = e.results[i];
        const { transcript, confidence } = risultato[0];
        if (risultato.isFinal) {
          if (confidence > 0 && confidence < FIDUCIA_MINIMA) incerto = true;
          suTesto(transcript.trim());
        } else {
          provvisorio += transcript;
        }
      }
      suAvviso(provvisorio ? `Ascolto… «${provvisorio.trim()}»` : "Ascolto…");
    };

    r.onerror = (e) => {
      if (ERRORI_DEL_SERVIZIO.has(e.error) && puoRegistrare()) {
        setSoloServer(true);
        suAvviso("Il riconoscimento del browser non risponde: premi di nuovo il microfono per registrare.");
      } else if (e.error === "not-allowed") {
        suAvviso("Il microfono non è autorizzato per questa pagina.");
      } else if (e.error === "no-speech") {
        suAvviso("Non ho sentito niente.");
      } else if (e.error !== "aborted") {
        suAvviso("La dettatura si è interrotta.");
      }
    };

    r.onend = () => {
      riconoscimento.current = null;
      setStato("fermo");
      if (incerto) suAvviso("Alcune parole erano incerte: rileggi prima di mandare.");
    };

    riconoscimento.current = r;
    suAvviso("Ascolto…");
    setStato("ascolta");
    r.start();
  };

  const registra = async () => {
    let flusso: MediaStream;
    try {
      flusso = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      suAvviso("Il microfono non è autorizzato per questa pagina.");
      return;
    }

    const formato = formatoDiRegistrazione();
    const r = new MediaRecorder(flusso, formato ? { mimeType: formato } : undefined);
    const pezzi: Blob[] = [];
    r.ondataavailable = (e) => { if (e.data.size) pezzi.push(e.data); };

    r.onstop = async () => {
      flusso.getTracks().forEach((t) => t.stop());
      if (scadenza.current) clearTimeout(scadenza.current);
      registratore.current = null;
      if (!token || pezzi.length === 0) {
        setStato("fermo");
        suAvviso(null);
        return;
      }
      setStato("trascrive");
      suAvviso("Trascrivo…");
      try {
        const esito = await trascriviRegistrazione(token, new Blob(pezzi, { type: r.mimeType }));
        if (esito.testo) suTesto(esito.testo.trim());
        suAvviso(
          !esito.testo
            ? "Non ho capito niente: riprova, più vicino al microfono."
            : esito.da_confermare
              ? "La trascrizione è incerta: rileggi prima di mandare."
              : null,
        );
      } catch (e) {
        suAvviso(e instanceof ErroreApi ? e.message : "Non sono riuscito a trascrivere.");
      } finally {
        setStato("fermo");
      }
    };

    registratore.current = r;
    r.start();
    setStato("registra");
    suAvviso("Registro… premi di nuovo per finire.");
    scadenza.current = setTimeout(() => {
      if (r.state === "recording") r.stop();
    }, SECONDI_MASSIMI * 1000);
  };

  const premi = () => {
    if (stato === "ascolta") {
      riconoscimento.current?.stop();
      return;
    }
    if (stato === "registra") {
      registratore.current?.stop();
      return;
    }
    if (stato === "trascrive") return;
    if (Browser) ascoltaNelBrowser(Browser);
    else if (puoRegistrare()) void registra();
  };

  const attivo = stato === "ascolta" || stato === "registra";
  const etichetta = !disponibile
    ? "La dettatura non è disponibile in questo browser"
    : attivo
      ? "Smetti di dettare"
      : stato === "trascrive"
        ? "Trascrizione in corso"
        : "Detta la domanda";

  return (
    <button
      type="button"
      className={attivo ? stili.attivo : stili.detta}
      onClick={premi}
      disabled={!disponibile || (disabilitato && !attivo) || stato === "trascrive"}
      aria-label={etichetta}
      aria-pressed={attivo}
      title={etichetta}
    >
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
        <rect x="6" y="1.5" width="6" height="10" rx="3" fill={attivo ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.5" />
        <path d="M3.5 8.5a5.5 5.5 0 0 0 11 0M9 14v2.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
      </svg>
    </button>
  );
}
