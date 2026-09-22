"use client";

import Link from "next/link";
import { use, useState } from "react";
import {
  archiviaPersonalita,
  collegaCorpus,
  creaVersione,
  dettaglioPersonalita,
  elencoBasi,
  elencoTipi,
  impostaTipi,
  opzioniRealizzazione,
  pubblicaVersione,
  scollegaCorpus,
  type Opzione,
  type PersonalitaDettaglio,
  type Versione,
  elencoAvatar,
  volgiAvatar,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import stili from "../../comuni.module.css";
import propri from "./dettaglio.module.css";

/* La lista dei siti si scrive una per riga, ma chi incolla da un foglio
 * separa con le virgole: si accettano entrambe invece di correggere l'utente. */
const NUOVA_RIGA = "\n";
const SEPARATORI = /[\n,]/;

export default function Dettaglio({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  /* In Next 16 i parametri di rotta sono una promise: `use()` la scioglie nel
     componente client. */
  const { id } = use(params);
  const { dati, errore, inCorso, ricarica } = useDati(
    (t) => dettaglioPersonalita(t, id),
    [id],
  );

  if (errore) return <p className={stili.errore}>{errore}</p>;
  if (inCorso && !dati) return <p className={stili.caricamento}>Caricamento…</p>;
  if (!dati) return null;

  return (
    <>
      <div className={stili.intestazione}>
        <div>
          <Link href="/" className={propri.ritorno}>
            ← Personalità
          </Link>
          <h1 className={stili.titolo}>{dati.display_name}</h1>
          <p className={stili.sottotitolo}>
            <span className={stili.mono}>{dati.slug}</span>
            {dati.description ? ` · ${dati.description}` : ""}
          </p>
        </div>
      </div>

      <Versioni personalita={dati} ricarica={ricarica} />
      <Corpora personalita={dati} ricarica={ricarica} />
      <Tipi personalita={dati} ricarica={ricarica} />
      <Volto personalita={dati} ricarica={ricarica} />
      <Realizzazione />
      <Archiviazione personalita={dati} ricarica={ricarica} />
    </>
  );
}

/* ---- versioni ---- */

function Versioni({
  personalita,
  ricarica,
}: {
  personalita: PersonalitaDettaglio;
  ricarica: () => void;
}) {
  const { esegui, inCorso, errore } = useAzione();
  const [aperta, setAperta] = useState(false);
  const corrente = personalita.versioni.find((v) => v.corrente);

  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>
        Versioni
        <button
          className={stili.secondaria}
          onClick={() => setAperta((v) => !v)}
        >
          {aperta ? "Annulla" : "Nuova versione"}
        </button>
      </h2>
      <p className={stili.riquadroNota}>
        Una versione pubblicata non si modifica: se ne crea un&apos;altra. Le
        conversazioni registrano quale versione le ha prodotte, ed è ciò che le
        rende riproducibili dopo mesi.
      </p>

      {aperta && (
        <ModuloVersione
          iniziale={corrente}
          alTermine={() => {
            setAperta(false);
            ricarica();
          }}
          personalitaId={personalita.id}
        />
      )}

      {errore && <p className={stili.errore}>{errore}</p>}

      {personalita.versioni.length === 0 ? (
        <div className={stili.vuoto}>
          Nessuna versione: questa personalità non può ancora rispondere.
        </div>
      ) : (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr>
                <th>N.</th>
                <th>Prompt</th>
                <th>Verifica</th>
                <th>Pubblicata</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {personalita.versioni.map((v) => (
                <tr key={v.id}>
                  <td className={stili.numero}>{v.version}</td>
                  <td className={propri.estratto}>
                    {v.system_prompt.slice(0, 160)}
                    {v.system_prompt.length > 160 ? "…" : ""}
                  </td>
                  <td className={stili.mono}>
                    {(v.guard_config?.groundcheck as string) ?? "citations"}
                  </td>
                  <td className={stili.mono}>
                    {v.published_at
                      ? new Date(v.published_at).toLocaleDateString("it-IT")
                      : "—"}
                  </td>
                  <td>
                    {v.corrente ? (
                      <span className={`${stili.stato} ${stili.pubblicata}`}>
                        ● in uso
                      </span>
                    ) : (
                      <button
                        className={stili.secondaria}
                        disabled={inCorso}
                        onClick={async () => {
                          await esegui((t) =>
                            pubblicaVersione(t, personalita.id, v.id),
                          );
                          ricarica();
                        }}
                      >
                        Pubblica
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

function ModuloVersione({
  personalitaId,
  iniziale,
  alTermine,
}: {
  personalitaId: string;
  iniziale?: Versione;
  alTermine: () => void;
}) {
  const { esegui, inCorso, errore } = useAzione();
  /* Si parte dalla versione in uso: quasi sempre si cambia un dettaglio, e
     riscrivere il prompt da capo invita a perdere qualcosa per strada. */
  const [prompt, setPrompt] = useState(iniziale?.system_prompt ?? "");
  const [verifica, setVerifica] = useState(
    (iniziale?.guard_config?.groundcheck as string) ?? "citations",
  );
  const [passaggi, setPassaggi] = useState(
    String((iniziale?.rag_config?.max_chunks as number) ?? 6),
  );

  /* La ricerca online sta qui e non fra le impostazioni della piattaforma
     perché è una proprietà della voce: Seneca non ha bisogno di sapere cosa
     è successo stamattina, una voce che commenta l'attualità sì. */
  const ricercaIniziale = (iniziale?.rag_config?.ricerca_online ?? {}) as {
    attiva?: boolean;
    siti?: string[];
    modo?: string;
    max_risultati?: number;
  };
  const [cerca, setCerca] = useState(Boolean(ricercaIniziale.attiva));
  const [siti, setSiti] = useState((ricercaIniziale.siti ?? []).join(NUOVA_RIGA));
  const [modoSiti, setModoSiti] = useState(
    ricercaIniziale.modo === "solo" ? "solo" : "anche",
  );
  const [risultati, setRisultati] = useState(
    String(ricercaIniziale.max_risultati ?? 3),
  );

  const listaSiti = siti
    .split(SEPARATORI)
    .map((s) => s.trim())
    .filter(Boolean);
  /* «Solo questi siti» con la lista vuota significa *nessun* sito, non
     *tutti*: dirlo qui evita una voce che tace e nessuno sa perché. */
  const listaMancante = cerca && modoSiti === "solo" && listaSiti.length === 0;

  const [pubblica, setPubblica] = useState(false);

  return (
    <div className={propri.modulo}>
      <div className={stili.campo}>
        <label className={stili.campoEtichetta} htmlFor="prompt">
          Prompt di sistema
        </label>
        <span className={stili.campoAiuto}>
          Entra nello strato stabile del prompt, quello riutilizzabile fra le
          richieste. Non metterci nulla che cambi da una domanda all&apos;altra:
          annullerebbe lo sconto sul contesto senza che nulla lo segnali.
        </span>
        <textarea
          id="prompt"
          className={stili.area}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      <div className={stili.riga}>
        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="verifica">
            Verifica di fondatezza
          </label>
          <select
            id="verifica"
            className={stili.selezione}
            value={verifica}
            onChange={(e) => setVerifica(e.target.value)}
          >
            <option value="citations">
              Citazioni — deterministica, costo zero
            </option>
            <option value="nli">
              Giudice — una chiamata in più per risposta
            </option>
          </select>
        </div>

        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="passaggi">
            Passaggi recuperati
          </label>
          <input
            id="passaggi"
            className={stili.ingresso}
            type="number"
            min={1}
            max={30}
            value={passaggi}
            onChange={(e) => setPassaggi(e.target.value)}
          />
        </div>
      </div>

      <fieldset className={propri.gruppo}>
        <legend className={propri.gruppoTitolo}>Ricerca online</legend>
        <p className={stili.campoAiuto}>
          Un corpus è chiuso per costruzione, ed è la sua virtù: si sa cosa c&apos;è
          dentro. Per una voce che commenta l&apos;attualità, o che risponde su un
          prodotto che si aggiorna, quella chiusura è il difetto. Ciò che arriva
          dal web entra in coda ai passaggi del corpus, con la sua fonte e la
          sua data, e la verifica di fondatezza lo tratta come gli altri.
        </p>

        <label className={propri.casella}>
          <input
            type="checkbox"
            checked={cerca}
            onChange={(e) => setCerca(e.target.checked)}
          />
          Cerca anche online
        </label>

        {cerca && (
          <>
            <div className={stili.campo}>
              <label className={stili.campoEtichetta} htmlFor="siti">
                Siti
              </label>
              <span className={stili.campoAiuto}>
                Uno per riga. Valgono anche i sottodomini: <code>example.com</code>{" "}
                comprende <code>docs.example.com</code>. Si può incollare
                l&apos;indirizzo completo.
              </span>
              <textarea
                id="siti"
                className={propri.areaCorta}
                value={siti}
                placeholder={`example.com${NUOVA_RIGA}docs.example.com`}
                onChange={(e) => setSiti(e.target.value)}
              />
            </div>

            <div className={stili.riga}>
              <div className={stili.campo}>
                <label className={stili.campoEtichetta} htmlFor="modo-siti">
                  Uso della lista
                </label>
                <select
                  id="modo-siti"
                  className={stili.selezione}
                  value={modoSiti}
                  onChange={(e) => setModoSiti(e.target.value)}
                >
                  <option value="anche">
                    Includili — ricerca aperta, questi privilegiati
                  </option>
                  <option value="solo">
                    Limitati a questi — nient&apos;altro entra
                  </option>
                </select>
              </div>

              <div className={stili.campo}>
                <label className={stili.campoEtichetta} htmlFor="risultati">
                  Pagine per risposta
                </label>
                <input
                  id="risultati"
                  className={stili.ingresso}
                  type="number"
                  min={1}
                  max={10}
                  value={risultati}
                  onChange={(e) => setRisultati(e.target.value)}
                />
              </div>
            </div>

            {listaMancante && (
              <p className={stili.errore}>
                «Limitati a questi» con la lista vuota significa nessuna fonte:
                la ricerca non verrebbe eseguita.
              </p>
            )}
          </>
        )}
      </fieldset>

      <label className={propri.casella}>
        <input
          type="checkbox"
          checked={pubblica}
          onChange={(e) => setPubblica(e.target.checked)}
        />
        Pubblica subito questa versione
      </label>

      {errore && <p className={stili.errore}>{errore}</p>}

      <button
        className={stili.primaria}
        disabled={inCorso || prompt.trim().length < 10 || listaMancante}
        onClick={async () => {
          const fatta = await esegui((t) =>
            creaVersione(t, personalitaId, {
              system_prompt: prompt.trim(),
              rag_config: {
                max_chunks: Number(passaggi) || 6,
                ricerca_online: {
                  attiva: cerca,
                  siti: listaSiti,
                  modo: modoSiti,
                  max_risultati: Number(risultati) || 3,
                },
              },
              guard_config: { groundcheck: verifica },
              pubblica,
            }),
          );
          if (fatta) alTermine();
        }}
      >
        {inCorso ? "Salvo…" : "Crea versione"}
      </button>
    </div>
  );
}

/* ---- corpora ---- */

function Corpora({
  personalita,
  ricarica,
}: {
  personalita: PersonalitaDettaglio;
  ricarica: () => void;
}) {
  const { dati: basi } = useDati(elencoBasi);
  const { esegui, inCorso, errore } = useAzione();
  const [scelta, setScelta] = useState("");
  const [ruolo, setRuolo] = useState("knowledge");

  const collegate = new Set(personalita.corpora.map((c) => c.kb_id));
  const disponibili = (basi ?? []).filter((b) => !collegate.has(b.id));
  const nomeDi = (id: string) =>
    basi?.find((b) => b.id === id)?.name ?? id.slice(0, 8);

  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>Corpora</h2>
      <p className={stili.riquadroNota}>
        <strong>Voce</strong> è il corpus da cui viene lo stile,{" "}
        <strong>conoscenza</strong> quello da cui vengono i fatti. La
        distinzione conta per la verifica: una frase in carattere non va
        misurata come se fosse un&apos;affermazione di fatto.
      </p>

      {personalita.corpora.length > 0 && (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr>
                <th>Base</th>
                <th>Ruolo</th>
                <th>Passaggi</th>
                <th>Attiva</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {personalita.corpora.map((c) => (
                <tr key={c.kb_id}>
                  <td>{nomeDi(c.kb_id)}</td>
                  <td className={stili.mono}>
                    {c.role === "voice" ? "voce" : "conoscenza"}
                  </td>
                  <td className={stili.numero}>{c.max_chunks}</td>
                  <td className={stili.mono}>{c.enabled ? "sì" : "no"}</td>
                  <td>
                    <button
                      className={stili.distruttiva}
                      disabled={inCorso}
                      onClick={async () => {
                        await esegui((t) =>
                          scollegaCorpus(t, personalita.id, c.kb_id),
                        );
                        ricarica();
                      }}
                    >
                      Scollega
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {errore && <p className={stili.errore}>{errore}</p>}

      {disponibili.length > 0 && (
        <div className={`${stili.riga} ${propri.aggiunta}`}>
          <select
            className={stili.selezione}
            value={scelta}
            onChange={(e) => setScelta(e.target.value)}
            aria-label="Base da collegare"
          >
            <option value="">Collega un corpus…</option>
            {disponibili.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name} ({b.stats?.passaggi ?? 0} passaggi)
              </option>
            ))}
          </select>
          <select
            className={stili.selezione}
            value={ruolo}
            onChange={(e) => setRuolo(e.target.value)}
            aria-label="Ruolo del corpus"
          >
            <option value="knowledge">conoscenza</option>
            <option value="voice">voce</option>
          </select>
          <button
            className={stili.secondaria}
            disabled={!scelta || inCorso}
            onClick={async () => {
              await esegui((t) =>
                collegaCorpus(t, personalita.id, {
                  kb_id: scelta,
                  role: ruolo,
                  max_chunks: 6,
                }),
              );
              setScelta("");
              ricarica();
            }}
          >
            Collega
          </button>
        </div>
      )}
    </div>
  );
}

/* ---- tipi ---- */

function Tipi({
  personalita,
  ricarica,
}: {
  personalita: PersonalitaDettaglio;
  ricarica: () => void;
}) {
  const { dati: tipi } = useDati(elencoTipi);
  const { esegui, inCorso } = useAzione();
  const attuali = new Set(personalita.tipi.map((t) => t.id));

  if (!tipi || tipi.length === 0) return null;

  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>Tipi</h2>
      <p className={stili.riquadroNota}>
        Una voce può essere più cose insieme: costringerla a sceglierne una
        perderebbe metà di ciò che la rende cercabile.
      </p>
      <div className={propri.caselle}>
        {tipi.map((t) => (
          <label key={t.id} className={propri.casella}>
            <input
              type="checkbox"
              checked={attuali.has(t.id)}
              disabled={inCorso}
              onChange={async (e) => {
                const nuovi = new Set(attuali);
                if (e.target.checked) nuovi.add(t.id);
                else nuovi.delete(t.id);
                await esegui((tok) =>
                  impostaTipi(tok, personalita.id, [...nuovi]),
                );
                ricarica();
              }}
            />
            {t.name}
          </label>
        ))}
      </div>
    </div>
  );
}

/* ---- realizzazione ---- */

/* ---- volto ---- */

/* Quale avatar indossa questa voce.
 *
 * Qui e non nella pagina degli avatar: la domanda che ci si pone guardando
 * una personalità è «che faccia ha?», e dover andare altrove per rispondere
 * — o per cambiarla — spezzerebbe il lavoro in due posti. Nessun volto è uno
 * stato normale: la voce si mostra col suo nome. */
function Volto({
  personalita,
  ricarica,
}: {
  personalita: PersonalitaDettaglio;
  ricarica: () => void;
}) {
  const { dati: avatar } = useDati(elencoAvatar);
  const { esegui, inCorso, errore } = useAzione();
  const [scelto, setScelto] = useState<string>(personalita.avatar_id ?? "");

  const attuale = avatar?.find((a) => a.id === personalita.avatar_id);

  return (
    <section className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>Volto</h2>
      <p className={stili.riquadroNota}>
        {attuale
          ? `Indossa «${attuale.name}» (${attuale.kind}${
              attuale.kind === "modello" ? ", con labiale" : ", senza labiale"
            }).`
          : "Nessun volto: la personalità si mostra col suo nome."}
      </p>

      {avatar && avatar.length === 0 ? (
        <p className={stili.riquadroNota}>
          Non ci sono avatar. Si creano nella pagina{" "}
          <Link href="/avatar">Avatar</Link>.
        </p>
      ) : (
        <div className={propri.rigaAzione}>
          <select
            className={stili.ingresso}
            value={scelto}
            onChange={(e) => setScelto(e.target.value)}
            aria-label="Avatar"
          >
            <option value="">Nessun volto</option>
            {(avatar ?? []).map((a) => (
              <option key={a.id} value={a.id}>
                {a.name} — {a.kind}
              </option>
            ))}
          </select>
          <button
            className={stili.primaria}
            disabled={inCorso || scelto === (personalita.avatar_id ?? "")}
            onClick={async () => {
              const fatto = await esegui((t) =>
                volgiAvatar(t, personalita.id, scelto || null),
              );
              if (fatto) ricarica();
            }}
          >
            {inCorso ? "Salvo…" : "Applica"}
          </button>
        </div>
      )}

      {errore && <p className={stili.errore}>{errore}</p>}
    </section>
  );
}

function Realizzazione() {
  const { dati } = useDati(opzioniRealizzazione);
  if (!dati) return null;

  const voci: [string, Opzione][] = [
    ...Object.entries(dati.modi),
    ...Object.entries(dati.esportazioni),
    ...Object.entries(dati.contesto),
  ];

  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>Realizzazione</h2>
      <p className={stili.riquadroNota}>
        Cosa questa installazione sa fare. Le voci spente non sono nascoste: il
        motivo accanto dice cosa manca, e un comando che sparisce sembrerebbe
        un difetto dell&apos;interfaccia.
      </p>
      <ul className={propri.opzioni}>
        {voci.map(([chiave, o]) => (
          <li
            key={chiave}
            className={o.available ? propri.opzione : propri.opzioneSpenta}
          >
            <label className={propri.casella}>
              <input type="checkbox" disabled={!o.available} />
              <span>{o.label}</span>
            </label>
            {!o.available && o.reason && (
              <p className={propri.motivo}>{o.reason}</p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ---- archiviazione ---- */

function Archiviazione({
  personalita,
  ricarica,
}: {
  personalita: PersonalitaDettaglio;
  ricarica: () => void;
}) {
  const { esegui, inCorso } = useAzione();
  const [conferma, setConferma] = useState(false);

  if (personalita.status === "archived") return null;

  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>Archiviazione</h2>
      <p className={stili.riquadroNota}>
        Toglie la voce dal catalogo. Non cancella nulla: le conversazioni già
        avvenute si riferiscono alle sue versioni, e portarle via le renderebbe
        illeggibili.
      </p>
      {conferma ? (
        <div className={stili.riga}>
          <button
            className={stili.distruttiva}
            disabled={inCorso}
            onClick={async () => {
              await esegui((t) => archiviaPersonalita(t, personalita.id));
              ricarica();
            }}
          >
            Confermo, archivia
          </button>
          <button
            className={stili.secondaria}
            onClick={() => setConferma(false)}
          >
            Annulla
          </button>
        </div>
      ) : (
        <button className={stili.secondaria} onClick={() => setConferma(true)}>
          Archivia
        </button>
      )}
    </div>
  );
}
