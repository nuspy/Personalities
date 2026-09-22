"use client";

import {
  assegnaModello,
  modelliECompiti,
  type CompitoAssegnato,
  type ModelloConfigurato,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import { Motore } from "../motore";
import comuni from "../comuni.module.css";
import stili from "./modelli.module.css";

/* Chi serve quale compito.
 *
 * Qui si sceglie fra nomi, non si scrivono indirizzi: l'elenco dei modelli
 * sta nella configurazione del servizio, e questa pagina non può ampliarlo.
 * La ragione è che un endpoint modificabile dall'interfaccia è traffico
 * dirottabile verso una macchina qualunque, con le chiavi appresso.
 */

const RICADUTA = "";

export default function Modelli() {
  const { dati, errore, inCorso, ricarica } = useDati(modelliECompiti);
  const { esegui, inCorso: inAzione, errore: erroreAzione } = useAzione();

  const cambia = async (compito: string, scelto: string) => {
    const fatto = await esegui((t) =>
      assegnaModello(t, compito, scelto === RICADUTA ? null : scelto),
    );
    if (fatto) ricarica();
  };

  return (
    <>
      <div className={comuni.intestazione}>
        <div>
          <h1 className={comuni.titolo}>Modelli</h1>
          <p className={comuni.sottotitolo}>
            Un solo modello per tutto è comodo finché i compiti non divergono,
            e divergono presto. Classificare cinquemila passaggi vuole un
            modello economico e instancabile; rispondere a un utente vuole il
            migliore che ci si possa permettere; verificare una risposta vuole
            un modello <em>diverso da quello che l&apos;ha scritta</em>.
          </p>
        </div>
      </div>

      <Motore />

      {errore && <p className={comuni.errore}>{errore}</p>}
      {erroreAzione && <p className={comuni.errore}>{erroreAzione}</p>}
      {inCorso && !dati && <p className={comuni.caricamento}>Caricamento…</p>}

      {dati && (
        <>
          <div className={stili.compiti}>
            {dati.compiti.map((c) => (
              <Scheda
                key={c.compito}
                compito={c}
                modelli={dati.modelli}
                bloccato={inAzione}
                onCambia={(scelto) => cambia(c.compito, scelto)}
              />
            ))}
          </div>

          <Elenco modelli={dati.modelli} />
        </>
      )}
    </>
  );
}

function Scheda({
  compito,
  modelli,
  bloccato,
  onCambia,
}: {
  compito: CompitoAssegnato;
  modelli: ModelloConfigurato[];
  bloccato: boolean;
  onCambia: (scelto: string) => void;
}) {
  const id = `compito-${compito.compito}`;

  return (
    <section className={stili.scheda}>
      <h2 className={stili.schedaTitolo}>{compito.label}</h2>
      <p className={stili.schedaNota}>{compito.descrizione}</p>

      <label className={stili.etichetta} htmlFor={id}>
        Modello
      </label>
      <select
        id={id}
        className={stili.scelta}
        value={compito.assegnato ?? RICADUTA}
        disabled={bloccato}
        onChange={(e) => onCambia(e.target.value)}
      >
        {/* Distinta da «assegnato al predefinito»: si comportano uguale oggi
            e diversamente domani, quando il predefinito cambia. */}
        <option value={RICADUTA}>
          Nessuna scelta — segue il predefinito
        </option>
        {modelli.map((m) => (
          <option key={m.nome} value={m.nome}>
            {m.nome}
            {m.senza_filtri ? " (senza filtri)" : ""}
          </option>
        ))}
      </select>

      <p className={stili.inUso}>
        In uso: <strong>{compito.in_uso}</strong>
        {compito.senza_filtri && (
          <>
            {" — "}
            <span className={stili.avviso}>
              modello senza filtri: non rifiuta nulla
            </span>
          </>
        )}
      </p>
    </section>
  );
}

function Elenco({ modelli }: { modelli: ModelloConfigurato[] }) {
  return (
    <section className={stili.disponibili}>
      <h2 className={stili.disponibiliTitolo}>Modelli configurati</h2>
      <p className={stili.disponibiliNota}>
        L&apos;elenco si cambia in <code>PERSONA_MODELLI</code>, non da qui:
        indirizzi e chiavi restano nella configurazione del servizio. Da questa
        pagina si sceglie fra questi nomi.
      </p>

      <div className={comuni.contenitoreTabella}>
        <table className={comuni.tabella}>
          <thead>
            <tr>
              <th>Nome</th>
              <th>Fornitore</th>
              <th>Modello</th>
              <th>Dove</th>
            </tr>
          </thead>
          <tbody>
            {modelli.map((m) => (
              <tr key={m.nome}>
                <td>
                  <strong>{m.nome}</strong>
                  {m.senza_filtri && (
                    <div className={stili.avviso}>senza filtri</div>
                  )}
                  {m.descrizione && (
                    <div className={comuni.campoAiuto}>{m.descrizione}</div>
                  )}
                </td>
                <td className={comuni.mono}>{m.provider}</td>
                <td className={comuni.mono}>{m.model || "—"}</td>
                <td className={comuni.mono}>{m.dove || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
