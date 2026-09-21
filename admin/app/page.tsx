"use client";

import Link from "next/link";
import { useState } from "react";
import {
  creaPersonalita,
  elencoPersonalita,
  type Personalita,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import stili from "./comuni.module.css";

export default function Personalita_() {
  const { dati, errore, inCorso, ricarica } = useDati(elencoPersonalita);
  const [apertoNuovo, setApertoNuovo] = useState(false);

  return (
    <>
      <div className={stili.intestazione}>
        <div>
          <h1 className={stili.titolo}>Personalità</h1>
          <p className={stili.sottotitolo}>
            Ogni voce ha una storia di versioni: quella pubblicata è la sola
            servita, e le precedenti restano perché le conversazioni già
            avvenute vi si riferiscono.
          </p>
        </div>
        <button
          className={stili.primaria}
          onClick={() => setApertoNuovo((v) => !v)}
        >
          {apertoNuovo ? "Annulla" : "Nuova personalità"}
        </button>
      </div>

      {apertoNuovo && (
        <ModuloNuova
          alTermine={() => {
            setApertoNuovo(false);
            ricarica();
          }}
        />
      )}

      {errore && <p className={stili.errore}>{errore}</p>}
      {inCorso && <p className={stili.caricamento}>Caricamento…</p>}

      {dati && dati.length === 0 && (
        <div className={stili.vuoto}>
          Nessuna personalità. Creane una per cominciare.
        </div>
      )}

      {dati && dati.length > 0 && <Tabella personalita={dati} />}
    </>
  );
}

function Tabella({ personalita }: { personalita: Personalita[] }) {
  return (
    <div className={stili.contenitoreTabella}>
      <table className={stili.tabella}>
        <thead>
          <tr>
            <th>Nome</th>
            <th>Slug</th>
            <th>Stato</th>
            <th>Tipi</th>
            <th>Categoria</th>
          </tr>
        </thead>
        <tbody>
          {personalita.map((p) => (
            <tr key={p.id}>
              <td>
                <Link href={`/personalita/${p.id}`} className={stili.collegamento}>
                  {p.display_name}
                </Link>
                {p.description && (
                  <div className={stili.campoAiuto}>{p.description}</div>
                )}
              </td>
              <td className={stili.mono}>{p.slug}</td>
              <td>
                <Stato stato={p.status} haVersione={!!p.current_version_id} />
              </td>
              <td>
                {p.tipi.length === 0 ? (
                  <span className={stili.campoAiuto}>—</span>
                ) : (
                  p.tipi.map((t) => (
                    <span key={t.id} className={stili.etichetta}>
                      {t.name}
                    </span>
                  ))
                )}
              </td>
              <td className={stili.mono}>{p.categoria?.name ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Stato({
  stato,
  haVersione,
}: {
  stato: Personalita["status"];
  haVersione: boolean;
}) {
  if (stato === "published") {
    return (
      <span className={`${stili.stato} ${stili.pubblicata}`}>
        <span aria-hidden="true">●</span> pubblicata
      </span>
    );
  }
  if (stato === "archived") {
    return (
      <span className={`${stili.stato} ${stili.archiviata}`}>archiviata</span>
    );
  }
  return (
    <span className={`${stili.stato} ${stili.bozza}`}>
      <span aria-hidden="true">○</span>{" "}
      {/* La distinzione conta: una bozza senza versione non potrebbe
          rispondere nemmeno se venisse pubblicata. */}
      {haVersione ? "bozza" : "bozza, senza versione"}
    </span>
  );
}

function ModuloNuova({ alTermine }: { alTermine: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [slug, setSlug] = useState("");
  const [nome, setNome] = useState("");
  const [descrizione, setDescrizione] = useState("");

  const slugValido = /^[a-z0-9][a-z0-9-]*$/.test(slug);

  return (
    <div className={stili.riquadro}>
      <h2 className={stili.riquadroTitolo}>Nuova personalità</h2>
      <p className={stili.riquadroNota}>
        Nasce in bozza: potrà rispondere solo quando avrà una versione
        pubblicata.
      </p>

      <div className={stili.riga}>
        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="nome">
            Nome
          </label>
          <input
            id="nome"
            className={stili.ingresso}
            value={nome}
            onChange={(e) => {
              setNome(e.target.value);
              /* Lo slug si propone dal nome finché nessuno lo tocca: è quasi
                 sempre quello giusto, e scriverlo due volte è lavoro inutile. */
              if (!slug || slug === proponiSlug(nome)) {
                setSlug(proponiSlug(e.target.value));
              }
            }}
            placeholder="Seneca"
          />
        </div>

        <div className={stili.campo}>
          <label className={stili.campoEtichetta} htmlFor="slug">
            Slug
          </label>
          <input
            id="slug"
            className={stili.ingresso}
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
            placeholder="seneca"
            aria-invalid={slug !== "" && !slugValido}
          />
          {slug !== "" && !slugValido && (
            <span className={stili.campoAiuto}>
              Solo minuscole, cifre e trattini.
            </span>
          )}
        </div>
      </div>

      <div className={stili.campo}>
        <label className={stili.campoEtichetta} htmlFor="descrizione">
          Descrizione
        </label>
        <input
          id="descrizione"
          className={stili.ingresso}
          value={descrizione}
          onChange={(e) => setDescrizione(e.target.value)}
          placeholder="Come si presenta nel catalogo"
        />
      </div>

      {errore && <p className={stili.errore}>{errore}</p>}

      <button
        className={stili.primaria}
        disabled={inCorso || !nome.trim() || !slugValido}
        onClick={async () => {
          const creata = await esegui((t) =>
            creaPersonalita(t, {
              slug,
              display_name: nome.trim(),
              description: descrizione.trim() || undefined,
            }),
          );
          if (creata) alTermine();
        }}
      >
        {inCorso ? "Creo…" : "Crea"}
      </button>
    </div>
  );
}

function proponiSlug(nome: string): string {
  return nome
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
