"use client";

import { useState } from "react";
import { registro, type VoceRegistro } from "@/lib/api";
import { useDati } from "@/lib/usa";
import stili from "../comuni.module.css";

/* Il registro delle operazioni.
 *
 * Sola lettura, e non per scelta dell'interfaccia: il backend non espone
 * alcun modo di modificarlo. Un registro che si può correggere non è un
 * registro.
 */
export default function Registro() {
  const [filtro, setFiltro] = useState("");
  const { dati, errore, inCorso } = useDati(
    (t) => registro(t, filtro || undefined),
    [filtro],
  );

  const azioni = [...new Set((dati ?? []).map((v) => v.action))].sort();

  return (
    <>
      <div className={stili.intestazione}>
        <div>
          <h1 className={stili.titolo}>Registro</h1>
          <p className={stili.sottotitolo}>
            Chi ha fatto cosa, a cosa, e com&apos;era prima. Ogni operazione
            amministrativa lascia una riga: la scrittura avviene dentro
            l&apos;operazione stessa, non accanto.
          </p>
        </div>
        <select
          className={stili.selezione}
          value={filtro}
          onChange={(e) => setFiltro(e.target.value)}
          aria-label="Filtra per azione"
          style={{ maxWidth: "16rem" }}
        >
          <option value="">Tutte le azioni</option>
          {azioni.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      </div>

      {errore && <p className={stili.errore}>{errore}</p>}
      {inCorso && <p className={stili.caricamento}>Caricamento…</p>}

      {dati && dati.length === 0 && (
        <div className={stili.vuoto}>Nessuna operazione registrata.</div>
      )}

      {dati && dati.length > 0 && (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr>
                <th>Quando</th>
                <th>Azione</th>
                <th>Oggetto</th>
                <th>Cambiamento</th>
                <th>Da</th>
              </tr>
            </thead>
            <tbody>
              {dati.map((v) => (
                <tr key={v.id}>
                  <td className={stili.mono}>
                    {new Date(v.created_at).toLocaleString("it-IT", {
                      day: "2-digit",
                      month: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </td>
                  <td className={stili.mono}>{v.action}</td>
                  <td className={stili.mono}>
                    {v.target_type}
                    {v.target_id ? ` ${v.target_id.slice(0, 8)}` : ""}
                  </td>
                  <td>
                    <Cambiamento voce={v} />
                  </td>
                  <td className={stili.mono}>{v.ip ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

/* Solo i campi cambiati.
 *
 * Mostrare i due stati per intero renderebbe la tabella illeggibile, e la
 * domanda a cui si risponde guardando un registro è «cosa è cambiato», non
 * «com'era tutto».
 */
function Cambiamento({ voce }: { voce: VoceRegistro }) {
  const prima = voce.before ?? {};
  const dopo = voce.after ?? {};
  const chiavi = [...new Set([...Object.keys(prima), ...Object.keys(dopo)])];
  const cambiate = chiavi.filter(
    (k) => JSON.stringify(prima[k]) !== JSON.stringify(dopo[k]),
  );

  if (cambiate.length === 0) {
    return <span className={stili.campoAiuto}>—</span>;
  }

  return (
    <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
      {cambiate.slice(0, 4).map((k) => (
        <li key={k} className={stili.mono} style={{ fontSize: "0.75rem" }}>
          {k}: {rendi(prima[k])} → {rendi(dopo[k])}
        </li>
      ))}
    </ul>
  );
}

function rendi(valore: unknown): string {
  if (valore === undefined || valore === null) return "—";
  if (typeof valore === "string") {
    return valore.length > 24 ? `${valore.slice(0, 24)}…` : valore;
  }
  return JSON.stringify(valore).slice(0, 30);
}
