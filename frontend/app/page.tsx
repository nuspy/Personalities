"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "react-oidc-context";
import {
  conversa,
  elencaPersonalita,
  ErroreApi,
  leggiCapacita,
  type CapacitaPiattaforma,
  type Consumo,
  type Fonte,
  type Personalita,
} from "@/lib/api";
import stili from "./page.module.css";

interface Turno {
  chi: "utente" | "voce";
  testo: string;
  ora: string;
  fonti?: Fonte[];
  consumo?: Consumo | null;
  errore?: string;
  inventati?: string[];
}

function adesso(): string {
  return new Date().toLocaleTimeString("it-IT", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function Pagina() {
  const auth = useAuth();

  if (auth.isLoading) return <Attesa />;
  if (auth.error) return <Accesso errore={auth.error.message} />;
  if (!auth.isAuthenticated) return <Accesso />;

  return <Conversazione />;
}

/* ------------------------------------------------------------------ */

function Attesa() {
  return (
    <div className={stili.pagina}>
      <div className={stili.accesso}>
        <p className={stili.nota}>Verifica della sessione…</p>
      </div>
    </div>
  );
}

function Accesso({ errore }: { errore?: string }) {
  const auth = useAuth();
  const [capacita, setCapacita] = useState<CapacitaPiattaforma | null>(null);
  /* `signinRedirect` restituisce una promise: se fallisce — servizio di
   * identità irraggiungibile, client configurato male — il rifiuto non
   * raccolto lascerebbe il pulsante apparentemente inerte, che è il guasto
   * più difficile da diagnosticare per chi lo subisce. */
  const [guasto, setGuasto] = useState<string | null>(null);

  useEffect(() => {
    /* Le capacità si leggono prima del login perché non richiedono identità e
     * perché dicono che cosa questa installazione sa fare: su un server senza
     * acceleratore l'addestramento non c'è, e vale la pena saperlo prima di
     * entrare invece di trovare un pulsante spento dopo. */
    leggiCapacita().then(setCapacita).catch(() => setCapacita(null));
  }, []);

  return (
    <div className={stili.pagina}>
      <header className={stili.testata}>
        <div className={stili.marchio}>
          Personalities<span>.</span>
        </div>
      </header>

      <main className={stili.accesso}>
        <div className={stili.accessoColonna}>
          <h1 className={stili.accessoTitolo}>
            Una voce ricostruita dai suoi <em>documenti</em>.
          </h1>
          <p className={stili.accessoTesto}>
            Ogni personalità nasce dal suo corpus: il lessico, le figure
            ricorrenti, il modo di argomentare. Quello che leggerai resta
            ancorato a ciò che è stato scritto.
          </p>

          <button
            className={stili.entra}
            onClick={() =>
              auth
                .signinRedirect()
                .catch((e) =>
                  setGuasto(
                    `Non riesco a raggiungere il servizio di accesso: ${
                      (e as Error).message
                    }`,
                  ),
                )
            }
          >
            Entra
          </button>

          {(errore || guasto) && (
            <p className={stili.errore}>{errore ?? guasto}</p>
          )}

          {capacita && <StatoInstallazione capacita={capacita} />}
        </div>
      </main>
    </div>
  );
}

/* Cosa sa fare questa installazione, in una riga.
 *
 * Il motivo per cui una funzione manca è informazione preziosa, ma non qui:
 * sulla porta d'ingresso sarebbe un muro di testo su cose che l'utente non ha
 * ancora chiesto. Il posto del motivo è accanto al comando disabilitato che lo
 * riguarda — la scheda «Realizzazione» della console, nella fase 3. Qui basta
 * dire in che modo il servizio risponde. */
function StatoInstallazione({ capacita }: { capacita: CapacitaPiattaforma }) {
  const attive = Object.values(capacita.features).filter((f) => f.available);
  const acceleratore = capacita.workers.find((w) => w.role === "gpu");

  return (
    <div className={stili.statoServizio}>
      <span className={stili.capacita}>
        <span
          className={attive.length ? stili.spia : stili.spiaSpenta}
          aria-hidden="true"
        >
          {attive.length ? "●" : "○"}
        </span>
        <span>
          {acceleratore
            ? `${acceleratore.gpu} — addestramento e modelli propri disponibili`
            : "Risposte ancorate ai documenti; nessun acceleratore collegato"}
        </span>
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Conversazione() {
  const auth = useAuth();
  const [turni, setTurni] = useState<Turno[]>([]);
  const [bozza, setBozza] = useState("");
  const [inCorso, setInCorso] = useState(false);
  const [pensa, setPensa] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [personalita, setPersonalita] = useState<Personalita[]>([]);
  const [scelta, setScelta] = useState<string | null>(null);

  const fondo = useRef<HTMLDivElement>(null);
  const campo = useRef<HTMLTextAreaElement>(null);
  const interruttore = useRef<AbortController | null>(null);

  useEffect(() => {
    elencaPersonalita()
      .then((elenco) => {
        setPersonalita(elenco);
        /* Se ce n'è una sola, si sceglie da sé: un menu con un'opzione è una
         * domanda a cui esiste una risposta sola. */
        if (elenco.length === 1) setScelta(elenco[0].slug);
      })
      .catch(() => setPersonalita([]));
  }, []);

  useEffect(() => {
    fondo.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turni]);

  const invia = useCallback(async () => {
    const messaggio = bozza.trim();
    if (!messaggio || inCorso) return;

    const token = auth.user?.access_token;
    if (!token) {
      auth.signinRedirect();
      return;
    }

    setBozza("");
    setInCorso(true);
    setTurni((precedenti) => [
      ...precedenti,
      { chi: "utente", testo: messaggio, ora: adesso() },
      { chi: "voce", testo: "", ora: adesso() },
    ]);

    const controller = new AbortController();
    interruttore.current = controller;

    const aggiornaUltimo = (modifica: (t: Turno) => Turno) =>
      setTurni((precedenti) => {
        const copia = [...precedenti];
        copia[copia.length - 1] = modifica(copia[copia.length - 1]);
        return copia;
      });

    try {
      for await (const evento of conversa(
        token,
        messaggio,
        conversationId,
        scelta,
        controller.signal,
      )) {
        switch (evento.tipo) {
          case "inizio":
            setConversationId(evento.conversationId);
            break;
          case "fonti":
            aggiornaUltimo((t) => ({ ...t, fonti: evento.fonti }));
            break;
          case "pensa":
            setPensa(true);
            break;
          case "token":
            setPensa(false);
            aggiornaUltimo((t) => ({ ...t, testo: t.testo + evento.testo }));
            break;
          case "degradato":
            aggiornaUltimo((t) => ({ ...t, errore: evento.motivo }));
            break;
          case "errore":
            aggiornaUltimo((t) => ({ ...t, errore: evento.messaggio }));
            break;
          case "fine":
            aggiornaUltimo((t) => ({
              ...t,
              consumo: evento.consumo,
              inventati: evento.inventati,
            }));
            break;
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        aggiornaUltimo((t) => ({ ...t, errore: "Interrotta." }));
      } else if (e instanceof ErroreApi && e.stato === 401) {
        auth.signinRedirect();
      } else {
        aggiornaUltimo((t) => ({
          ...t,
          errore: (e as Error).message || "Il servizio non ha risposto.",
        }));
      }
    } finally {
      setInCorso(false);
      setPensa(false);
      interruttore.current = null;
      campo.current?.focus();
    }
  }, [auth, bozza, conversationId, inCorso, scelta]);

  const nomeScelta =
    personalita.find((p) => p.slug === scelta)?.display_name ?? null;

  return (
    <div className={stili.pagina}>
      <header className={stili.testata}>
        <div className={stili.marchio}>
          Personalities<span>.</span>
        </div>

        {personalita.length > 1 && (
          <select
            className={stili.scelta}
            value={scelta ?? ""}
            /* La personalità si fissa all'apertura della conversazione:
             * cambiarla a metà renderebbe lo scambio incoerente, e l'API la
             * ignorerebbe comunque. */
            disabled={inCorso || turni.length > 0}
            onChange={(e) => setScelta(e.target.value || null)}
            aria-label="Con chi parlare"
          >
            <option value="">Nessuna personalità</option>
            {personalita.map((p) => (
              <option key={p.slug} value={p.slug}>
                {p.display_name}
              </option>
            ))}
          </select>
        )}

        <button
          className={stili.azioneTestata}
          onClick={() => {
            setTurni([]);
            setConversationId(null);
            campo.current?.focus();
          }}
          disabled={inCorso || turni.length === 0}
        >
          Nuova
        </button>
        <button
          className={stili.azioneTestata}
          onClick={() => auth.signoutRedirect()}
        >
          Esci
        </button>
      </header>

      <main className={stili.lettura}>
        <div className={stili.colonna}>
          {turni.length === 0 ? (
            <Soglia
              nome={auth.user?.profile.given_name}
              personalita={nomeScelta}
            />
          ) : (
            turni.map((turno, i) => (
              <Riga
                key={i}
                turno={turno}
                inScrittura={
                  inCorso && i === turni.length - 1 && turno.chi === "voce"
                }
                pensa={pensa && i === turni.length - 1}
              />
            ))
          )}
          <div ref={fondo} />
        </div>
      </main>

      <div className={stili.composer}>
        <div className={stili.composerColonna}>
          <textarea
            ref={campo}
            className={stili.campo}
            value={bozza}
            rows={1}
            placeholder="Scrivi la tua domanda"
            aria-label="La tua domanda"
            onChange={(e) => {
              setBozza(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
            onKeyDown={(e) => {
              /* Invio manda, Maiusc+Invio va a capo. Su touch la scorciatoia
               * non si applica: lì il tasto Invio deve andare a capo, perché
               * mandare per sbaglio è irreversibile. */
              if (e.key === "Enter" && !e.shiftKey && !("ontouchstart" in window)) {
                e.preventDefault();
                invia();
              }
            }}
          />
          {inCorso ? (
            <button
              className={stili.ferma}
              onClick={() => interruttore.current?.abort()}
              aria-label="Interrompi la risposta"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                <rect width="14" height="14" fill="currentColor" />
              </svg>
            </button>
          ) : (
            <button
              className={stili.invia}
              onClick={invia}
              disabled={!bozza.trim()}
              aria-label="Manda la domanda"
            >
              <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
                <path
                  d="M9 15V3M9 3L4 8M9 3l5 5"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  fill="none"
                  strokeLinecap="square"
                  transform="rotate(180 9 9)"
                />
              </svg>
            </button>
          )}
        </div>
        <p className={stili.suggerimento}>
          <span className={stili.scorciatoia}>Invio per mandare</span>
          <span>{nomeScelta ?? "senza personalità"}</span>
        </p>
      </div>
    </div>
  );
}

function Soglia({
  nome,
  personalita,
}: {
  nome?: string;
  personalita: string | null;
}) {
  return (
    <div className={stili.soglia}>
      <h1 className={stili.sogliaTitolo}>
        {personalita
          ? `Scrivi a ${personalita}.`
          : nome
            ? `Bentornato, ${nome}.`
            : "Comincia."}
      </h1>
      <p className={stili.sogliaTesto}>
        {personalita
          ? "Le risposte nascono dal suo corpus. A margine trovi i passaggi da cui vengono."
          : "Il modello risponde con la propria voce: scegli una personalità per ancorare le risposte a un corpus."}
      </p>
      <p className={stili.nota}>
        Quello che scrivi resta salvato nella conversazione e legato al tuo
        account.
      </p>
    </div>
  );
}

function Riga({
  turno,
  inScrittura,
  pensa,
}: {
  turno: Turno;
  inScrittura: boolean;
  pensa: boolean;
}) {
  const èVoce = turno.chi === "voce";

  return (
    <article
      className={`${stili.turno} ${èVoce ? "" : stili.turnoUtente}`}
      aria-live={inScrittura ? "polite" : undefined}
    >
      {/* Il margine dell'apparato: chi parla, quando, e da dove viene ciò che
          dice. I riferimenti compaiono appena il recupero è finito, quindi
          prima che la risposta cominci ad arrivare. */}
      <div className={stili.margine}>
        {èVoce && (
          <span className={stili.segno} aria-hidden="true">
            ¶
          </span>
        )}
        <span className={stili.chi}>{èVoce ? "voce" : "tu"}</span>
        <span className={stili.ora}>{turno.ora}</span>
        {turno.fonti && turno.fonti.length > 0 && (
          <Apparato fonti={turno.fonti} />
        )}
      </div>

      <div>
        <div className={stili.testo}>
          {turno.testo}
          {inScrittura && !turno.errore && (
            <span className={stili.cursore} aria-hidden="true" />
          )}
        </div>
        {pensa && !turno.testo && (
          <p className={stili.pensa}>sta ragionando…</p>
        )}
        {turno.errore && <p className={stili.errore}>{turno.errore}</p>}
        {turno.inventati && turno.inventati.length > 0 && (
          /* Un riferimento citato che non esiste fra quelli forniti. Si
             rileva confrontando due insiemi: nessun modello di verifica, costo
             zero. Va detto a chi legge, perché è l'unico punto in cui una
             risposta ben scritta può essere infondata. */
          <p className={stili.avviso}>
            Cita riferimenti che non esistono ({turno.inventati.join(", ")}):
            quella parte non è ancorata al corpus.
          </p>
        )}
      </div>
    </article>
  );
}

function Apparato({ fonti }: { fonti: Fonte[] }) {
  return (
    <ul className={stili.apparato}>
      {fonti.map((f) => (
        <li key={f.etichetta} className={stili.riferimento}>
          <abbr
            className={stili.etichetta}
            title={`${f.documento}${f.sezione ? ` — ${f.sezione}` : ""}\n\n${f.estratto}`}
          >
            {f.etichetta}
          </abbr>{" "}
          <span className={stili.fonte}>{f.documento}</span>
        </li>
      ))}
    </ul>
  );
}
