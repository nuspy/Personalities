"use client";

/* I volti delle personalità.
 *
 * **La validazione avviene al salvataggio, e il messaggio arriva qui.** Un
 * avatar rotto scoperto mentre qualcuno conversa è un riquadro vuoto e un
 * errore nella console del browser di un utente: nessuna informazione per chi
 * l'ha configurato, e nessun modo di collegarla alla modifica che l'ha
 * causata. Il server rifiuta e dice cosa manca; questa pagina lo mostra
 * accanto al campo, mentre chi ha sbagliato ha ancora in mano il perché.
 *
 * **La configurazione si compila per campi, non a mano in JSON.** I tre tipi
 * vogliono cose diverse — un indirizzo, una clip per stato, quindici pose —
 * e un'area di testo libera sposterebbe sull'utente il lavoro di ricordare la
 * forma esatta, che è precisamente quello che il server poi rifiuta.
 */

import { useCallback, useState } from "react";
import {
  creaAvatar,
  elencoAvatar,
  eliminaAvatar,
  modificaAvatar,
  type Avatar,
} from "@/lib/api";
import { useAzione, useDati } from "@/lib/usa";
import comuni from "../comuni.module.css";
import stili from "./avatar.module.css";

/* Le forme della bocca, nell'ordine in cui il server le dichiara.
 *
 * Copiate e non chieste all'API: sono quindici stringhe che cambiano solo se
 * cambia la tassonomia dei visemi, e una chiamata in più all'apertura della
 * pagina per ottenerle costerebbe più di quanto renda. Il server resta
 * l'autorità — se diverge, rifiuta il salvataggio nominando la forma ignota. */
const VISEMI = [
  "A", "E", "I", "O", "U",
  "MBP", "FV", "L", "TH", "SS", "CH", "KG", "NN", "RR", "X",
];

const DESCRIZIONE: Record<string, string> = {
  immagine:
    "Un ritratto fermo. Nessun labiale: mentre la voce parla, accanto al volto si anima un'onda sonora.",
  video:
    "Clip in ciclo, una per stato. La bocca si muove nei momenti giusti senza formare le parole giuste.",
  modello:
    "Un file .glb con i morph target. È l'unico che segue i visemi forma per forma.",
};

const STATI_VIDEO = ["fermo", "parlante", "ascolto"] as const;

export default function Avatars() {
  const { dati, errore, inCorso, ricarica } = useDati(elencoAvatar);
  const [aperto, setAperto] = useState(false);
  const [inModifica, setInModifica] = useState<Avatar | null>(null);

  return (
    <>
      <div className={comuni.intestazione}>
        <div>
          <h1 className={comuni.titolo}>Avatar</h1>
          <p className={comuni.sottotitolo}>
            Il server non disegna mai: manda l&apos;audio e i tempi delle forme
            della bocca, e ad animare è il browser di chi ascolta. Un avatar si
            riusa fra più personalità, e non è versionato col prompt — un
            ritratto sostituito non cambia il senso delle conversazioni
            passate.
          </p>
        </div>
        <button
          className={comuni.primaria}
          onClick={() => {
            setInModifica(null);
            setAperto((v) => !v);
          }}
        >
          {aperto ? "Annulla" : "Nuovo avatar"}
        </button>
      </div>

      {(aperto || inModifica) && (
        <Modulo
          avatar={inModifica}
          alTermine={() => {
            setAperto(false);
            setInModifica(null);
            ricarica();
          }}
        />
      )}

      {errore && <p className={comuni.errore}>{errore}</p>}
      {inCorso && !dati && <p className={comuni.caricamento}>Caricamento…</p>}

      {dati && dati.length === 0 && (
        <div className={comuni.vuoto}>
          Nessun avatar. Una personalità senza volto si mostra col suo nome:
          è lo stato normale, non un difetto.
        </div>
      )}

      {dati && dati.length > 0 && (
        <div className={stili.griglia}>
          {dati.map((a) => (
            <Scheda
              key={a.id}
              avatar={a}
              suModifica={() => {
                setAperto(false);
                setInModifica(a);
              }}
              suElimina={ricarica}
            />
          ))}
        </div>
      )}
    </>
  );
}

/* ---- scheda ---- */

function Scheda({
  avatar,
  suModifica,
  suElimina,
}: {
  avatar: Avatar;
  suModifica: () => void;
  suElimina: () => void;
}) {
  const { esegui, inCorso } = useAzione();
  const [conferma, setConferma] = useState(false);
  const [esito, setEsito] = useState<string | null>(null);

  const uri = String(avatar.config?.uri ?? "");
  const pose = (avatar.config?.pose ?? {}) as Record<string, string>;
  const mancanti = VISEMI.filter((v) => !(v in pose));

  return (
    <article className={stili.scheda}>
      <div className={stili.anteprima}>
        {avatar.kind === "immagine" && uri ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img className={stili.miniatura} src={uri} alt="" />
        ) : (
          <span className={stili.simbolo} aria-hidden="true">
            {avatar.kind === "video" ? "▶" : avatar.kind === "modello" ? "◈" : "◯"}
          </span>
        )}
      </div>

      <div className={stili.corpo}>
        <h2 className={stili.nome}>{avatar.name}</h2>
        <p className={comuni.mono}>{avatar.slug}</p>
        <p className={stili.tipo}>
          <span className={stili.etichettaTipo}>{avatar.kind}</span>
          {avatar.kind === "modello" ? (
            <span className={stili.labiale}>labiale</span>
          ) : (
            <span className={stili.senzaLabiale}>senza labiale</span>
          )}
        </p>

        {avatar.description && (
          <p className={stili.descrizione}>{avatar.description}</p>
        )}

        {avatar.kind === "modello" && mancanti.length > 0 && (
          /* Un modello con dieci pose su quindici anima peggio, non male: le
             forme mancanti il client le rende con la più vicina, e chi ha
             configurato l'avatar deve saperlo invece di scoprirlo guardando. */
          <p className={stili.avviso}>
            {mancanti.length} pose mancanti: {mancanti.join(" ")}
          </p>
        )}

        {esito && <p className={stili.avviso}>{esito}</p>}

        <div className={stili.azioni}>
          <button className={comuni.secondaria} onClick={suModifica}>
            Modifica
          </button>
          {conferma ? (
            <button
              className={comuni.distruttiva}
              disabled={inCorso}
              onClick={async () => {
                const r = await esegui((t) => eliminaAvatar(t, avatar.id));
                if (!r) return;
                if (r.personalita_senza_volto > 0) {
                  setEsito(
                    `${r.personalita_senza_volto} personalità sono rimaste senza volto.`,
                  );
                }
                suElimina();
              }}
            >
              Confermi?
            </button>
          ) : (
            <button
              className={comuni.distruttiva}
              onClick={() => setConferma(true)}
            >
              Elimina
            </button>
          )}
        </div>
      </div>
    </article>
  );
}

/* ---- modulo ---- */

function Modulo({
  avatar,
  alTermine,
}: {
  avatar: Avatar | null;
  alTermine: () => void;
}) {
  const { esegui, inCorso, errore } = useAzione();

  const [slug, setSlug] = useState(avatar?.slug ?? "");
  const [nome, setNome] = useState(avatar?.name ?? "");
  const [tipo, setTipo] = useState(avatar?.kind ?? "immagine");
  const [descrizione, setDescrizione] = useState(avatar?.description ?? "");
  const [uri, setUri] = useState(String(avatar?.config?.uri ?? ""));
  const [poster, setPoster] = useState(String(avatar?.config?.poster ?? ""));
  const [clip, setClip] = useState<Record<string, string>>(
    (avatar?.config?.clip as Record<string, string>) ?? {},
  );
  const [pose, setPose] = useState<Record<string, string>>(
    (avatar?.config?.pose as Record<string, string>) ?? {},
  );

  const configurazione = useCallback((): Record<string, unknown> => {
    if (tipo === "video") {
      const pulite = Object.fromEntries(
        Object.entries(clip).filter(([, v]) => v.trim()),
      );
      return { clip: pulite };
    }
    if (tipo === "modello") {
      const pulite = Object.fromEntries(
        Object.entries(pose).filter(([, v]) => v.trim()),
      );
      return { uri, pose: pulite, ...(poster ? { poster } : {}) };
    }
    return { uri };
  }, [clip, pose, poster, tipo, uri]);

  const salva = async () => {
    const config = configurazione();
    const fatto = avatar
      ? await esegui((t) =>
          modificaAvatar(t, avatar.id, {
            name: nome,
            kind: tipo,
            config,
            description: descrizione || null,
          }),
        )
      : await esegui((t) =>
          creaAvatar(t, {
            slug,
            name: nome,
            kind: tipo,
            config,
            description: descrizione || undefined,
          }),
        );
    if (fatto) alTermine();
  };

  return (
    <section className={comuni.riquadro}>
      <h2 className={comuni.riquadroTitolo}>
        {avatar ? `Modifica «${avatar.name}»` : "Nuovo avatar"}
      </h2>

      <div className={stili.riga}>
        {!avatar && (
          <label className={comuni.campo}>
            <span className={comuni.campoEtichetta}>Slug</span>
            <input
              className={comuni.ingresso}
              value={slug}
              onChange={(e) => setSlug(e.target.value)}
              placeholder="ritratto-seneca"
            />
          </label>
        )}
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Nome</span>
          <input
            className={comuni.ingresso}
            value={nome}
            onChange={(e) => setNome(e.target.value)}
            placeholder="Ritratto di Seneca"
          />
        </label>
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>Tipo</span>
          <select
            className={comuni.ingresso}
            value={tipo}
            onChange={(e) => setTipo(e.target.value as Avatar["kind"])}
          >
            <option value="immagine">Immagine ferma</option>
            <option value="video">Video in ciclo</option>
            <option value="modello">Modello 3D</option>
          </select>
        </label>
      </div>

      <p className={comuni.riquadroNota}>{DESCRIZIONE[tipo]}</p>

      {tipo !== "video" && (
        <label className={comuni.campo}>
          <span className={comuni.campoEtichetta}>
            {tipo === "modello" ? "Indirizzo del .glb" : "Indirizzo dell'immagine"}
          </span>
          <input
            className={comuni.ingresso}
            value={uri}
            onChange={(e) => setUri(e.target.value)}
            placeholder="https://…"
          />
        </label>
      )}

      {tipo === "video" && (
        <div className={stili.riga}>
          {STATI_VIDEO.map((stato) => (
            <label key={stato} className={comuni.campo}>
              <span className={comuni.campoEtichetta}>
                Clip «{stato}»{stato === "fermo" ? " — obbligatoria" : ""}
              </span>
              <input
                className={comuni.ingresso}
                value={clip[stato] ?? ""}
                onChange={(e) =>
                  setClip((c) => ({ ...c, [stato]: e.target.value }))
                }
                placeholder="https://…"
              />
            </label>
          ))}
        </div>
      )}

      {tipo === "modello" && (
        <>
          <label className={comuni.campo}>
            <span className={comuni.campoEtichetta}>
              Immagine di ripiego (facoltativa)
            </span>
            <input
              className={comuni.ingresso}
              value={poster}
              onChange={(e) => setPoster(e.target.value)}
              placeholder="https://…"
            />
            <span className={comuni.campoAiuto}>
              Si vede mentre il modello scende, e resta se il browser non sa
              disegnarlo.
            </span>
          </label>

          <h3 className={comuni.riquadroTitolo}>Pose</h3>
          <p className={comuni.riquadroNota}>
            Il nome del morph target dentro il file, per ciascuna forma della
            bocca. I nomi variano fra strumenti —{" "}
            <code>viseme_PP</code>, <code>mouthPucker</code>,{" "}
            <code>01_Ah</code> — e indovinarli darebbe un volto immobile senza
            errori da nessuna parte. <strong>X</strong> è la bocca a riposo ed
            è obbligatoria: senza, una pausa lascerebbe il volto congelato
            nell&apos;ultima forma pronunciata.
          </p>

          <div className={stili.pose}>
            {VISEMI.map((forma) => (
              <label key={forma} className={stili.posa}>
                <span
                  className={
                    forma === "X" ? stili.formaRichiesta : stili.forma
                  }
                >
                  {forma}
                </span>
                <input
                  className={stili.ingressoPosa}
                  value={pose[forma] ?? ""}
                  onChange={(e) =>
                    setPose((p) => ({ ...p, [forma]: e.target.value }))
                  }
                  placeholder={`viseme_${forma}`}
                />
              </label>
            ))}
          </div>
        </>
      )}

      <label className={comuni.campo}>
        <span className={comuni.campoEtichetta}>Descrizione</span>
        <input
          className={comuni.ingresso}
          value={descrizione}
          onChange={(e) => setDescrizione(e.target.value)}
        />
      </label>

      {errore && <p className={comuni.errore}>{errore}</p>}

      <div className={stili.azioni}>
        <button className={comuni.primaria} disabled={inCorso} onClick={salva}>
          {inCorso ? "Salvo…" : avatar ? "Salva" : "Crea"}
        </button>
        <button className={comuni.secondaria} onClick={alTermine}>
          Chiudi
        </button>
      </div>
    </section>
  );
}
