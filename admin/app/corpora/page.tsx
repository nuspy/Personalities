"use client";

import { useState } from "react";
import {
  creaBase,
  documentiDi,
  elencoBasi,
  eliminaDocumento,
  type Base,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import stili from "../comuni.module.css";

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

      {dati && dati.length === 0 && (
        <div className={stili.vuoto}>
          Nessun documento. Si aggiungono con{" "}
          <code className={stili.mono}>platform_core.tools.seed_seneca</code> o
          dall&apos;ingestione, che arriva con la fase 3b.
        </div>
      )}

      {dati && dati.length > 0 && (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr>
                <th>Titolo</th>
                <th>Origine</th>
                <th>Impronta</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {dati.map((d) => (
                <tr key={d.id}>
                  <td>{d.title}</td>
                  <td className={stili.mono}>
                    {d.uri ? (
                      <a
                        href={d.uri}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={stili.collegamento}
                      >
                        {new URL(d.uri).hostname}
                      </a>
                    ) : (
                      "—"
                    )}
                  </td>
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
