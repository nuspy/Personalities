"use client";

/* Esperimenti fra versioni della stessa personalità.
 *
 * Ogni utente riceve sempre la stessa variante, assegnata dall'impronta del
 * suo identificativo: niente da configurare per chi parla, e un confronto
 * che misura le versioni invece del disorientamento di chi le vede
 * alternarsi. La misura è il voto sulle risposte; il verdetto arriva solo
 * oltre un minimo di voti per variante, perché con pochi voti una
 * differenza vistosa è quasi sempre il caso.
 */

import { useCallback, useState } from "react";
import {
  apriEsperimento,
  concludiEsperimento,
  dettaglioEsperimento,
  dettaglioPersonalita,
  elencoEsperimenti,
  elencoPersonalita,
  type Esperimento,
  type RisultatoVariante,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import comuni from "../comuni.module.css";
import { Barre, percento } from "../grafici";
import stili from "./esperimenti.module.css";

const data = new Intl.DateTimeFormat("it-IT", { day: "numeric", month: "short", year: "numeric" });

const plurale = (n: number, uno: string, molti: string) => `${n} ${n === 1 ? uno : molti}`;

export default function Esperimenti() {
  const { dati, errore, inCorso, ricarica } = useDati(elencoEsperimenti);
  const [aperto, setAperto] = useState<string | null>(null);
  const [nuovo, setNuovo] = useState(false);

  return (
    <>
      <div className={comuni.intestazione}>
        <div>
          <h1 className={comuni.titolo}>Esperimenti</h1>
          <p className={comuni.sottotitolo}>
            Due o più versioni della stessa personalità servite a utenti
            diversi, sempre la stessa alla stessa persona. Si confrontano sui
            voti alle risposte; il verdetto arriva solo con abbastanza voti.
          </p>
        </div>
        <button className={comuni.primaria} onClick={() => setNuovo((v) => !v)}>
          {nuovo ? "Annulla" : "Nuovo esperimento"}
        </button>
      </div>

      {nuovo && (
        <NuovoEsperimento
          alTermine={() => {
            setNuovo(false);
            ricarica();
          }}
        />
      )}

      {errore && <p className={comuni.errore}>{errore}</p>}
      {inCorso && !dati && <p className={comuni.caricamento}>Caricamento…</p>}
      {dati && dati.length === 0 && (
        <div className={comuni.vuoto}>
          Nessun esperimento. Servono due versioni della stessa personalità:
          creane una nuova dalla sua scheda, poi confrontala qui con quella
          pubblicata.
        </div>
      )}

      {dati && dati.length > 0 && (
        <ul className={stili.elenco}>
          {dati.map((e) => (
            <li key={e.id} className={stili.voce}>
              <button
                className={stili.voceTesta}
                onClick={() => setAperto(aperto === e.id ? null : e.id)}
                aria-expanded={aperto === e.id}
              >
                <span className={e.stato === "attivo" ? stili.statoAttivo : stili.statoConcluso}>
                  {e.stato}
                </span>
                <span className={stili.voceNome}>{e.nome}</span>
                <span className={stili.voceDettagli}>
                  {e.personalita?.nome} · {e.varianti.map((v) => `${v.etichetta}: v${v.versione ?? "?"}`).join(" / ")} ·
                  dal {data.format(new Date(e.iniziato_il))}
                </span>
              </button>
              {aperto === e.id && <Dettaglio id={e.id} alCambio={ricarica} />}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

function Dettaglio({ id, alCambio }: { id: string; alCambio: () => void }) {
  const { dati, errore, ricarica } = useDati(useCallback((t: string) => dettaglioEsperimento(t, id), [id]));

  if (errore) return <p className={comuni.errore}>{errore}</p>;
  if (!dati) return <p className={comuni.caricamento}>Caricamento…</p>;

  const varianti = dati.risultati.varianti;

  return (
    <div className={stili.dettaglio}>
      {dati.ipotesi && (
        <p className={stili.ipotesi}>
          <strong>Ipotesi.</strong> {dati.ipotesi}
        </p>
      )}

      <Barre
        titolo="Approvazione per variante"
        descrizione={`Risposte votate utili sul totale delle votate. Verdetto oltre ${dati.risultati.voti_minimi} voti per variante.`}
        unita="% utili"
        barre={varianti.map((v) => ({
          chiave: v.version_id,
          etichetta: `${v.etichetta} · v${dati.varianti.find((x) => x.version_id === v.version_id)?.versione ?? "?"}`,
          valore: v.approvazione === null ? null : Math.round(v.approvazione * 100),
          dettaglio: `${plurale(v.su + v.giu, "voto", "voti")} su ${plurale(v.risposte, "risposta", "risposte")} · ${plurale(v.utenti, "utente", "utenti")}`,
        }))}
      />

      <div className={comuni.contenitoreTabella}>
        <table className={comuni.tabella}>
          <thead>
            <tr>
              <th>Variante</th>
              <th className={comuni.numero}>Utenti</th>
              <th className={comuni.numero}>Risposte</th>
              <th className={comuni.numero}>Utili / non utili</th>
              <th className={comuni.numero}>Approvazione</th>
              <th className={comuni.numero}>Primo token</th>
              <th className={comuni.numero}>Citazioni inventate</th>
              <th>Contro il riferimento</th>
            </tr>
          </thead>
          <tbody>
            {varianti.map((v, i) => (
              <RigaVariante key={v.version_id} variante={v} riferimento={i === 0} />
            ))}
          </tbody>
        </table>
      </div>

      {dati.stato === "attivo" ? (
        <Conclusione
          esperimento={dati}
          alTermine={() => {
            ricarica();
            alCambio();
          }}
        />
      ) : (
        <p className={stili.esito}>
          Concluso il {dati.concluso_il ? data.format(new Date(dati.concluso_il)) : "—"}
          {dati.vincitore
            ? `: vince ${dati.varianti.find((v) => v.version_id === dati.vincitore)?.etichetta}.`
            : ", senza vincitore."}
          {dati.conclusione && <> {dati.conclusione}</>}
        </p>
      )}
    </div>
  );
}

function RigaVariante({ variante: v, riferimento }: { variante: RisultatoVariante; riferimento: boolean }) {
  const c = v.confronto;
  return (
    <tr>
      <td><strong>{v.etichetta}</strong>{riferimento && <span className={stili.riferimento}> riferimento</span>}</td>
      <td className={comuni.numero}>{v.utenti}</td>
      <td className={comuni.numero}>{v.risposte}</td>
      <td className={comuni.numero}>{v.su} / {v.giu}</td>
      <td className={comuni.numero}>{percento(v.approvazione)}</td>
      <td className={comuni.numero}>{v.primo_token_ms_p50 !== null ? `${v.primo_token_ms_p50} ms` : "—"}</td>
      <td className={comuni.numero}>{percento(v.citazioni_inventate, 1)}</td>
      <td>
        {riferimento ? "—" : c ? (
          <span>
            {c.differenza !== null && `${c.differenza > 0 ? "+" : ""}${(c.differenza * 100).toFixed(1).replace(".", ",")} punti · `}
            {c.verdetto}
            {c.p !== null && ` (p = ${c.p.toFixed(3).replace(".", ",")})`}
          </span>
        ) : "—"}
      </td>
    </tr>
  );
}

function Conclusione({ esperimento, alTermine }: { esperimento: Esperimento; alTermine: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [vincitore, setVincitore] = useState("");
  const [testo, setTesto] = useState("");
  const [pubblica, setPubblica] = useState(true);
  const [conferma, setConferma] = useState(false);

  return (
    <div className={comuni.riquadro} style={{ marginTop: "1rem" }}>
      <h3 className={comuni.riquadroTitolo}>Concludi</h3>
      <p className={comuni.riquadroNota}>
        Concluso, l&apos;esperimento smette di assegnare varianti: le conversazioni
        nuove ricevono la versione pubblicata. Pubblicare il vincitore lo rende
        quella versione per tutti.
      </p>
      <div className={comuni.riga}>
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Vincitore</span>
          <select className={comuni.selezione} value={vincitore} onChange={(e) => setVincitore(e.target.value)}>
            <option value="">Nessuno</option>
            {esperimento.varianti.map((v) => (
              <option key={v.version_id} value={v.version_id}>{v.etichetta} · versione {v.versione}</option>
            ))}
          </select>
        </label>
        <label className={comuni.campo} style={{ flex: "2 1 18rem" }}>
          <span className={comuni.campoEtichetta}>Che cosa si è imparato</span>
          <input className={comuni.ingresso} value={testo} onChange={(e) => setTesto(e.target.value)} />
        </label>
      </div>
      <label className={stili.spunta}>
        <input type="checkbox" checked={pubblica && !!vincitore} disabled={!vincitore} onChange={(e) => setPubblica(e.target.checked)} />
        Pubblica il vincitore
      </label>
      {errore && <p className={comuni.errore}>{errore}</p>}
      {conferma ? (
        <div className={comuni.riga}>
          <button
            className={comuni.distruttiva}
            disabled={inCorso}
            onClick={async () => {
              const fatto = await esegui((t) =>
                concludiEsperimento(t, esperimento.id, {
                  vincitore: vincitore || null,
                  conclusione: testo.trim() || undefined,
                  pubblica: pubblica && !!vincitore,
                }),
              );
              if (fatto) alTermine();
            }}
          >
            Confermo: concludi{vincitore && pubblica ? " e pubblica" : ""}
          </button>
          <button className={comuni.secondaria} onClick={() => setConferma(false)}>No</button>
        </div>
      ) : (
        <button className={comuni.secondaria} onClick={() => setConferma(true)}>Concludi l&apos;esperimento</button>
      )}
    </div>
  );
}

function NuovoEsperimento({ alTermine }: { alTermine: () => void }) {
  const { dati: personalita } = useDati(elencoPersonalita);
  const [scelta, setScelta] = useState("");
  const { dati: dettaglio } = useDati(
    useCallback((t: string) => (scelta ? dettaglioPersonalita(t, scelta) : Promise.resolve(null)), [scelta]),
  );
  const [nome, setNome] = useState("");
  const [ipotesi, setIpotesi] = useState("");
  const [pesi, setPesi] = useState<Record<string, number>>({});
  const { esegui, inCorso, errore } = useAzione();

  const versioni = dettaglio?.versioni ?? [];
  const scelte = Object.entries(pesi).filter(([, p]) => p > 0);
  /* La versione corrente in testa: è il riferimento, contro cui le altre
   * si confrontano. */
  const ordinate = [...scelte].sort(
    ([a], [b]) => Number(versioni.find((v) => v.id === b)?.corrente) - Number(versioni.find((v) => v.id === a)?.corrente),
  );

  return (
    <div className={comuni.riquadro}>
      <h2 className={comuni.riquadroTitolo}>Nuovo esperimento</h2>
      <p className={comuni.riquadroNota}>
        Scrivi l&apos;ipotesi prima di guardare i numeri: un esperimento senza
        ipotesi finisce per confermare quella che fa comodo. La versione
        pubblicata fa da riferimento.
      </p>

      <div className={comuni.riga}>
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Personalità</span>
          <select
            className={comuni.selezione}
            value={scelta}
            onChange={(e) => {
              setScelta(e.target.value);
              setPesi({});
            }}
          >
            <option value="">Scegli…</option>
            {personalita?.map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
          </select>
        </label>
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Nome</span>
          <input className={comuni.ingresso} value={nome} onChange={(e) => setNome(e.target.value)} placeholder="Risposte con un aneddoto" />
        </label>
      </div>
      <label className={comuni.campo}>
        <span className={comuni.campoEtichetta}>Ipotesi</span>
        <input
          className={comuni.ingresso}
          value={ipotesi}
          onChange={(e) => setIpotesi(e.target.value)}
          placeholder="Aprire con un aneddoto rende le risposte più apprezzate"
        />
      </label>

      {versioni.length > 0 && (
        <fieldset className={stili.versioni}>
          <legend className={comuni.campoEtichetta}>Versioni da confrontare, con il peso di ciascuna</legend>
          {versioni.map((v) => (
            <label key={v.id} className={stili.versione}>
              <input
                type="checkbox"
                checked={(pesi[v.id] ?? 0) > 0}
                onChange={(e) => setPesi((p) => ({ ...p, [v.id]: e.target.checked ? 50 : 0 }))}
              />
              <span>
                Versione {v.version}{v.corrente && " · pubblicata"}
                <span className={stili.anteprima}>{v.system_prompt.slice(0, 90)}…</span>
              </span>
              {(pesi[v.id] ?? 0) > 0 && (
                <input
                  type="number"
                  min={1}
                  max={100}
                  className={stili.peso}
                  aria-label={`Peso della versione ${v.version}`}
                  value={pesi[v.id]}
                  onChange={(e) => setPesi((p) => ({ ...p, [v.id]: Math.max(1, Math.min(100, Number(e.target.value) || 1)) }))}
                />
              )}
            </label>
          ))}
        </fieldset>
      )}
      {scelta && versioni.length < 2 && dettaglio && (
        <p className={comuni.riquadroNota}>Questa personalità ha una sola versione: creane un&apos;altra dalla sua scheda.</p>
      )}

      {errore && <p className={comuni.errore}>{errore}</p>}
      <button
        className={comuni.primaria}
        disabled={inCorso || !scelta || nome.trim().length < 3 || scelte.length < 2}
        onClick={async () => {
          const fatto = await esegui((t) =>
            apriEsperimento(t, scelta, {
              nome: nome.trim(),
              ipotesi: ipotesi.trim() || undefined,
              varianti: ordinate.map(([version_id, peso]) => ({ version_id, peso })),
            }),
          );
          if (fatto) alTermine();
        }}
      >
        {inCorso ? "Apro…" : "Apri l'esperimento"}
      </button>
    </div>
  );
}
