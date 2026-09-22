"use client";

/* Il banco di prova dei prompt.
 *
 * Si parte da una versione salvata, si cambia ciò che si vuole provare, e si
 * fa una domanda. La risposta arriva con il prompt esatto che l'ha prodotta e
 * i passaggi recuperati: il punto non è vedere *che cosa* risponde, ma
 * *perché* — e la sola risposta non lo dice.
 *
 * Niente di ciò che succede qui tocca il prodotto: nessuna conversazione,
 * nessun credito, nessuna memoria, nessuna versione nuova. Per portare una
 * modifica agli utenti la si salva come versione dalla scheda della
 * personalità, e la si confronta con la pubblicata in un esperimento.
 */

import { useCallback, useState } from "react";
import {
  dettaglioPersonalita,
  elencoPersonalita,
  provaPrompt,
  type EsitoPlayground,
  type Personalita,
  type Versione,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import comuni from "../comuni.module.css";
import stili from "./prove.module.css";

export default function Prove() {
  const { dati: personalita } = useDati(elencoPersonalita);
  const [scelta, setScelta] = useState("");
  const { dati: dettaglio } = useDati(
    useCallback((t: string) => (scelta ? dettaglioPersonalita(t, scelta) : Promise.resolve(null)), [scelta]),
  );
  const [versioneScelta, setVersioneScelta] = useState<string | null>(null);

  const versioni = dettaglio?.versioni ?? [];
  const versione =
    versioni.find((v) => v.id === versioneScelta) ?? versioni.find((v) => v.corrente) ?? versioni[0];

  return (
    <>
      <div className={comuni.intestazione}>
        <div>
          <h1 className={comuni.titolo}>Prove</h1>
          <p className={comuni.sottotitolo}>
            Cambia il prompt, le regole o il recupero e guarda come cambia la
            risposta, con il prompt esatto che è stato mandato. Nessuna
            conversazione, nessun credito, nessuna versione salvata.
          </p>
        </div>
      </div>

      <div className={comuni.riga}>
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Personalità</span>
          <select
            className={comuni.selezione}
            value={scelta}
            onChange={(e) => {
              setScelta(e.target.value);
              setVersioneScelta(null);
            }}
          >
            <option value="">Scegli…</option>
            {personalita?.map((p: Personalita) => (
              <option key={p.id} value={p.id}>{p.display_name}</option>
            ))}
          </select>
        </label>
        {versioni.length > 0 && versione && (
          <label className={comuni.campo}>
            <span className={comuni.campoEtichetta}>Versione di partenza</span>
            <select
              className={comuni.selezione}
              value={versione.id}
              onChange={(e) => setVersioneScelta(e.target.value)}
            >
              {versioni.map((v) => (
                <option key={v.id} value={v.id}>
                  Versione {v.version}{v.corrente ? " · pubblicata" : ""}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {scelta && dettaglio && versioni.length === 0 && (
        <div className={comuni.vuoto}>Questa personalità non ha ancora versioni da provare.</div>
      )}

      {/* `key`: cambiando versione il banco riparte dai suoi valori, invece
          di tenere le modifiche fatte su un'altra. */}
      {versione && <Banco key={versione.id} personalitaId={scelta} versione={versione} />}
    </>
  );
}

function regoleDi(v: Versione): string[] {
  const regole = (v.behavior_rules as { regole?: unknown } | null)?.regole;
  return Array.isArray(regole) ? regole.map(String) : [];
}

function Banco({ personalitaId, versione }: { personalitaId: string; versione: Versione }) {
  const temperaturaBase = Number((versione.llm_config as { temperature?: number } | null)?.temperature ?? 0.7);
  const passaggiBase = Number((versione.rag_config as { max_chunks?: number } | null)?.max_chunks ?? 6);

  const [prompt, setPrompt] = useState(versione.system_prompt);
  const [regole, setRegole] = useState(regoleDi(versione).join("\n"));
  const [temperatura, setTemperatura] = useState(temperaturaBase);
  const [passaggi, setPassaggi] = useState(passaggiBase);
  const [domanda, setDomanda] = useState("");
  const [confronta, setConfronta] = useState(true);
  const [esiti, setEsiti] = useState<{ salvata?: EsitoPlayground; modificata?: EsitoPlayground }>({});
  const { esegui, inCorso, errore } = useAzione();

  const listaRegole = regole.split("\n").map((r) => r.trim()).filter(Boolean);
  const modifiche = {
    ...(prompt !== versione.system_prompt ? { system_prompt: prompt } : {}),
    ...(listaRegole.join("\n") !== regoleDi(versione).join("\n") ? { regole: listaRegole } : {}),
    ...(temperatura !== temperaturaBase ? { temperature: temperatura } : {}),
    ...(passaggi !== passaggiBase ? { max_chunks: passaggi } : {}),
  };
  const modificata = Object.keys(modifiche).length > 0;

  const prova = async () => {
    const base = { personality_id: personalitaId, version_id: versione.id, domanda: domanda.trim() };
    setEsiti({});
    const risultato = await esegui(async (t) => {
      /* In parallelo: la stessa domanda alle due versioni nello stesso
       * momento, invece di far aspettare il doppio. */
      const [salvata, nuova] = await Promise.all([
        confronta && modificata ? provaPrompt(t, base) : Promise.resolve(undefined),
        provaPrompt(t, { ...base, ...modifiche }),
      ]);
      return { salvata, modificata: nuova };
    });
    if (risultato) setEsiti(risultato);
  };

  return (
    <div className={stili.banco}>
      <section className={comuni.riquadro}>
        <h2 className={comuni.riquadroTitolo}>Che cosa provare</h2>

        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Prompt di sistema</span>
          <textarea
            className={`${comuni.ingresso} ${stili.areaTesto}`}
            rows={8}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </label>

        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Regole di comportamento, una per riga</span>
          <textarea
            className={`${comuni.ingresso} ${stili.areaTesto}`}
            rows={4}
            value={regole}
            onChange={(e) => setRegole(e.target.value)}
          />
        </label>

        <div className={comuni.riga}>
          <label className={comuni.campo}>
            <span className={comuni.campoEtichetta}>Temperatura: {temperatura.toFixed(1).replace(".", ",")}</span>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.1}
              value={temperatura}
              onChange={(e) => setTemperatura(Number(e.target.value))}
              className={stili.cursore}
            />
          </label>
          <label className={comuni.campo}>
            <span className={comuni.campoEtichetta}>Passaggi recuperati</span>
            <input
              type="number"
              min={0}
              max={20}
              className={comuni.ingresso}
              value={passaggi}
              onChange={(e) => setPassaggi(Math.max(0, Math.min(20, Number(e.target.value) || 0)))}
            />
          </label>
        </div>

        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Domanda</span>
          <textarea
            className={`${comuni.ingresso} ${stili.areaTesto}`}
            rows={2}
            value={domanda}
            onChange={(e) => setDomanda(e.target.value)}
            placeholder="Che cosa pensi del tempo che passa?"
          />
        </label>

        <label className={stili.spunta}>
          <input
            type="checkbox"
            checked={confronta}
            disabled={!modificata}
            onChange={(e) => setConfronta(e.target.checked)}
          />
          Confronta con la versione {versione.version} così com&apos;è salvata
        </label>

        {errore && <p className={comuni.errore}>{errore}</p>}
        <div className={comuni.riga}>
          <button className={comuni.primaria} disabled={inCorso || !domanda.trim()} onClick={prova}>
            {inCorso ? "Il modello risponde…" : modificata ? "Prova la modifica" : "Prova"}
          </button>
          {modificata && (
            <span className={stili.modifiche}>
              Modificati: {Object.keys(modifiche).map((k) => ({
                system_prompt: "prompt", regole: "regole", temperature: "temperatura", max_chunks: "passaggi",
              }[k])).join(", ")}
            </span>
          )}
        </div>
      </section>

      {(esiti.salvata || esiti.modificata) && (
        <div className={esiti.salvata ? stili.affiancati : undefined}>
          {esiti.salvata && <Esito titolo={`Versione ${versione.version} salvata`} esito={esiti.salvata} />}
          {esiti.modificata && (
            <Esito
              titolo={esiti.modificata.versione.modificata ? "Con le modifiche" : `Versione ${versione.version}`}
              esito={esiti.modificata}
            />
          )}
        </div>
      )}
    </div>
  );
}

function Esito({ titolo, esito }: { titolo: string; esito: EsitoPlayground }) {
  const uso = esito.uso;
  return (
    <section className={stili.esito}>
      <h3 className={stili.esitoTitolo}>{titolo}</h3>
      <div className={stili.risposta}>{esito.risposta}</div>

      {esito.riferimenti_inventati.length > 0 && (
        <p className={comuni.errore}>
          Cita riferimenti che non esistono: {esito.riferimenti_inventati.join(", ")}.
        </p>
      )}

      <dl className={stili.misure}>
        <div><dt>Primo token</dt><dd>{esito.tempi_ms.primo_token ?? "—"} ms</dd></div>
        <div><dt>Totale</dt><dd>{esito.tempi_ms.totale} ms</dd></div>
        <div><dt>Token</dt><dd>{uso ? `${uso.prompt_tokens} + ${uso.completion_tokens}` : "—"}</dd></div>
        <div><dt>Da cache</dt><dd>{uso?.cached_tokens ?? 0}</dd></div>
        <div><dt>Strategia</dt><dd>{esito.strategia ? esito.strategia.nome : "—"}</dd></div>
      </dl>

      {esito.passaggi.length > 0 && (
        <details className={stili.piega}>
          <summary>Passaggi recuperati ({esito.passaggi.length})</summary>
          <ol className={stili.passaggi}>
            {esito.passaggi.map((p) => (
              <li key={p.etichetta}>
                <span className={comuni.mono}>[{p.etichetta}]</span> <strong>{p.documento}</strong>
                {p.sezione && <> · {p.sezione}</>}
                <div className={stili.estratto}>{p.estratto}</div>
              </li>
            ))}
          </ol>
        </details>
      )}

      <details className={stili.piega}>
        <summary>Prompt inviato ({esito.prompt.length} messaggi)</summary>
        {esito.prompt.map((m, i) => (
          <div key={i} className={stili.messaggio}>
            <div className={stili.ruolo}>
              {m.role}
              {esito.punto_di_cache !== null && i === esito.punto_di_cache && (
                <span className={stili.cache}> · fine del prefisso in cache</span>
              )}
            </div>
            <pre className={stili.contenuto}>{m.content}</pre>
          </div>
        ))}
      </details>
    </section>
  );
}
