"use client";

import { useState } from "react";
import {
  elencoBasi,
  provaRecupero,
  type EsitoProva,
  type PassaggioProvato,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import stili from "../comuni.module.css";
import propri from "./recupero.module.css";

/* L'ispezione del recupero.
 *
 * Quando una risposta è sbagliata, la domanda è sempre la stessa: il recupero
 * non ha trovato ciò che serviva, o l'ha trovato e scartato? Sono due difetti
 * con due rimedi opposti — nel primo caso si lavora sul corpus o sui
 * passaggi, nel secondo sull'ordinamento — e distinguerli richiede di vedere
 * il recupero senza la generazione in mezzo.
 */
export default function Recupero() {
  const { dati: basi } = useDati(elencoBasi);
  const { esegui, inCorso, errore } = useAzione();
  const [domanda, setDomanda] = useState("");
  const [scelte, setScelte] = useState<string[]>([]);
  const [limite, setLimite] = useState("6");
  const [esito, setEsito] = useState<EsitoProva | null>(null);

  return (
    <>
      <div className={stili.intestazione}>
        <div>
          <h1 className={stili.titolo}>Recupero</h1>
          <p className={stili.sottotitolo}>
            Interroga i corpora senza generare nulla. Vedi cosa sarebbe finito
            nel prompt, cosa è rimasto fuori, e con quali punteggi.
          </p>
        </div>
      </div>

      <div className={stili.riquadro}>
        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="domanda">
            Domanda
          </label>
          <input
            id="domanda"
            className={stili.ingresso}
            value={domanda}
            onChange={(e) => setDomanda(e.target.value)}
            placeholder="come si acquista la virtù?"
            onKeyDown={(e) => {
              if (e.key === "Enter" && domanda.trim() && scelte.length) {
                esegui((t) =>
                  provaRecupero(t, {
                    domanda: domanda.trim(),
                    kb_ids: scelte,
                    limite: Number(limite) || 6,
                  }),
                ).then((r) => r && setEsito(r));
              }
            }}
          />
        </div>

        <div className={stili.campo}>
          <span className={stili.campoEtichetta}>Corpora da interrogare</span>
          <div className={propri.caselle}>
            {(basi ?? []).map((b) => (
              <label key={b.id} className={propri.casella}>
                <input
                  type="checkbox"
                  checked={scelte.includes(b.id)}
                  onChange={(e) =>
                    setScelte((s) =>
                      e.target.checked
                        ? [...s, b.id]
                        : s.filter((x) => x !== b.id),
                    )
                  }
                />
                {b.name}
                <span className={stili.mono}>
                  ({b.stats?.passaggi ?? 0})
                </span>
              </label>
            ))}
          </div>
        </div>

        <div className={stili.riga}>
          <div className={stili.campo}>
            <label className={stili.campoEtichetta} htmlFor="limite">
              Passaggi da scegliere
            </label>
            <input
              id="limite"
              className={stili.ingresso}
              type="number"
              min={1}
              max={20}
              value={limite}
              onChange={(e) => setLimite(e.target.value)}
            />
          </div>
          <button
            className={stili.primaria}
            disabled={inCorso || !domanda.trim() || scelte.length === 0}
            onClick={async () => {
              const r = await esegui((t) =>
                provaRecupero(t, {
                  domanda: domanda.trim(),
                  kb_ids: scelte,
                  limite: Number(limite) || 6,
                }),
              );
              if (r) setEsito(r);
            }}
          >
            {inCorso ? "Cerco…" : "Cerca"}
          </button>
        </div>

        {errore && <p className={stili.errore}>{errore}</p>}
      </div>

      {esito && (
        <>
          <Elenco
            titolo="Scelti"
            nota="Questi sarebbero finiti nel prompt, con le etichette che il modello userebbe per citarli."
            passaggi={esito.scelti}
          />
          <Elenco
            titolo="Scartati"
            nota="Trovati ma rimasti fuori. Se ciò che serviva è qui, il problema è l'ordinamento e non il corpus."
            passaggi={esito.scartati}
            spenti
          />
        </>
      )}
    </>
  );
}

function Elenco({
  titolo,
  nota,
  passaggi,
  spenti = false,
}: {
  titolo: string;
  nota: string;
  passaggi: PassaggioProvato[];
  spenti?: boolean;
}) {
  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>
        {titolo}
        <span className={stili.mono}>{passaggi.length}</span>
      </h2>
      <p className={stili.riquadroNota}>{nota}</p>

      {passaggi.length === 0 ? (
        <div className={stili.vuoto}>Nessuno.</div>
      ) : (
        <ol className={propri.passaggi}>
          {passaggi.map((p, i) => (
            <li
              key={i}
              className={spenti ? propri.passaggioSpento : propri.passaggio}
            >
              <div className={propri.riga}>
                {p.etichetta && (
                  <span className={propri.etichetta}>[{p.etichetta}]</span>
                )}
                <span className={propri.documento}>
                  {p.documento}
                  {p.sezione ? ` — ${p.sezione}` : ""}
                </span>
                <span className={propri.punteggi}>
                  <Punteggio
                    nome="rrf"
                    valore={p.rrf.toFixed(4)}
                    forte
                  />
                  <Punteggio
                    nome="vett."
                    valore={
                      p.posizione_vettoriale
                        ? `#${p.posizione_vettoriale}`
                        : "—"
                    }
                  />
                  <Punteggio
                    nome="less."
                    valore={
                      p.posizione_lessicale ? `#${p.posizione_lessicale}` : "—"
                    }
                  />
                  {/* Trovato da entrambe le ricerche: due metodi indipendenti
                      che concordano sono un segnale più forte di uno solo che
                      insiste, ed è il motivo per cui l'ibrido esiste. */}
                  {p.trovato_da_entrambe && (
                    <span className={propri.concordi} title="trovato da entrambe le ricerche">
                      ⁑
                    </span>
                  )}
                </span>
              </div>
              <p className={propri.testo}>{p.testo}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function Punteggio({
  nome,
  valore,
  forte = false,
}: {
  nome: string;
  valore: string;
  forte?: boolean;
}) {
  return (
    <span className={forte ? propri.punteggioForte : propri.punteggio}>
      <span className={propri.nomePunteggio}>{nome}</span> {valore}
    </span>
  );
}
