"use client";

/* Il volto 3D, con la bocca guidata dai visemi.
 *
 * **Perché three.js e non un video.** Il server manda tempi, non fotogrammi:
 * un modello con i morph target può assumere quindici forme diverse guidato
 * da qualche kilobyte di dati, mentre un video parlante andrebbe generato per
 * ogni risposta e trasmesso per ogni ascoltatore.
 *
 * **Perché si interpola e non si scatta.** Una bocca che salta dalla forma
 * precedente a quella nuova a ogni cambio di visema — venti volte al secondo —
 * sembra un burattino. Le forme si mescolano su una finestra corta: il peso
 * della forma in corso sale, quello della precedente scende, e nel mezzo la
 * bocca passa per le posizioni intermedie come farebbe una vera.
 *
 * **Perché il caricamento è pigro.** three.js e il caricatore GLTF pesano
 * qualche centinaio di kilobyte: chi conversa con una voce senza volto — la
 * maggioranza — non deve scaricarli. Si importano alla prima comparsa di un
 * avatar di tipo `modello`.
 */

import { useEffect, useRef, useState } from "react";
import type { Mesh, Object3D, PerspectiveCamera } from "three";
import type { Volto } from "@/lib/api";
import stili from "./voce.module.css";

/* Quanto dura la fusione fra due forme, in secondi.
 *
 * Ottanta millisecondi: poco più della durata minima di un visema, così che
 * una forma faccia in tempo ad affermarsi prima che la successiva cominci a
 * toglierle peso. Più corto scatta, più lungo impasta le forme fino a
 * renderle indistinguibili — una bocca che si muove senza articolare. */
const FUSIONE = 0.08;

/* Quanto apre la bocca un visema al massimo del suo peso.
 *
 * I morph target dei modelli sono di solito tarati su 1 = espressione piena,
 * che nel parlato è esagerata: si articola con una frazione dell'escursione, e
 * a 1 il volto sembra masticare. */
const AMPIEZZA = 0.85;

interface Props {
  volto: Volto;
  /** La forma corrente, dalla riproduzione. `X` è la bocca a riposo. */
  forma: string;
  parlando: boolean;
}

interface Scena {
  smonta: () => void;
  imposta: (forma: string) => void;
}

export function Volto3D({ volto, forma, parlando }: Props) {
  const contenitore = useRef<HTMLDivElement>(null);
  const scena = useRef<Scena | null>(null);
  const [stato, setStato] = useState<"carico" | "pronto" | "guasto">("carico");
  const [motivo, setMotivo] = useState("");

  useEffect(() => {
    const nodo = contenitore.current;
    if (!nodo) return;

    let annullato = false;

    costruisci(nodo, volto)
      .then((s) => {
        if (annullato) {
          s.smonta();
          return;
        }
        scena.current = s;
        setStato("pronto");
      })
      .catch((e: unknown) => {
        if (annullato) return;
        /* Un modello che non si carica è un problema di chi ha configurato
         * l'avatar, non di chi sta conversando: si dice qui e si ripiega sul
         * poster, invece di lasciare un riquadro vuoto e un errore nella
         * console che nessuno leggerà. */
        setMotivo((e as Error)?.message ?? "modello non caricato");
        setStato("guasto");
      });

    return () => {
      annullato = true;
      scena.current?.smonta();
      scena.current = null;
    };
  }, [volto]);

  useEffect(() => {
    scena.current?.imposta(parlando ? forma : "X");
  }, [forma, parlando]);

  if (stato === "guasto") {
    const poster = (volto.extra?.poster as string | undefined) ?? "";
    return poster ? (
      // eslint-disable-next-line @next/next/no-img-element
      <img className={stili.ritratto} src={poster} alt={volto.nome} title={motivo} />
    ) : (
      <div className={stili.ritrattoAssente} role="img" aria-label={volto.nome} title={motivo}>
        {volto.nome.slice(0, 1)}
      </div>
    );
  }

  return (
    <div
      ref={contenitore}
      className={stili.ritratto3d}
      role="img"
      aria-label={volto.nome}
      data-stato={stato}
    />
  );
}

/* ---- la scena ---- */

async function costruisci(nodo: HTMLElement, volto: Volto): Promise<Scena> {
  const THREE = await import("three");
  const { GLTFLoader } = await import(
    "three/examples/jsm/loaders/GLTFLoader.js"
  );

  const lato = Math.max(nodo.clientWidth, 1);
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(lato, lato, false);
  nodo.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100);

  /* Due luci e non una: con la sola frontale il volto è piatto, e i morph
   * target della bocca — che sono rilievi di pochi millimetri — non
   * proiettano nulla e diventano invisibili. */
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.6));
  const direzionale = new THREE.DirectionalLight(0xffffff, 1.1);
  direzionale.position.set(1, 1.5, 2);
  scene.add(direzionale);

  const gltf = await new GLTFLoader().loadAsync(volto.uri);
  const modello = gltf.scene;
  scene.add(modello);

  const scala = Number(volto.extra?.scala ?? 1) || 1;
  modello.scale.setScalar(scala);

  inquadra(THREE, modello, camera, volto);

  /* Le pose, risolte una volta sola.
   *
   * La corrispondenza fra visema e morph target arriva dal descrittore perché
   * i nomi variano fra strumenti: `viseme_PP`, `mouthPucker`, `01_Ah`.
   * Risolverli a ogni fotogramma costerebbe una ricerca per stringa sessanta
   * volte al secondo per una mappa che non cambia mai. */
  const bersagli = risolviPose(modello, volto.pose);

  let corrente = "X";
  let precedente = "X";
  let daQuando = performance.now() / 1000;
  let vivo = true;

  const disegna = () => {
    if (!vivo) return;

    const t = performance.now() / 1000;
    const avanzamento = Math.min(1, (t - daQuando) / FUSIONE);

    for (const [forma, riferimenti] of bersagli) {
      let peso = 0;
      if (forma === corrente) peso = avanzamento;
      else if (forma === precedente) peso = 1 - avanzamento;
      if (forma === "X") peso *= 0.35;   // il riposo è una posa tenue

      for (const { mesh, indice } of riferimenti) {
        mesh.morphTargetInfluences![indice] = peso * AMPIEZZA;
      }
    }

    renderer.render(scene, camera);
    requestAnimationFrame(disegna);
  };
  requestAnimationFrame(disegna);

  /* Il riquadro può cambiare misura — rotazione del telefono, colonna che si
   * restringe — e un renderer che non se ne accorge disegna un volto
   * schiacciato finché non si ricarica la pagina. */
  const osservatore = new ResizeObserver(() => {
    const nuovo = Math.max(nodo.clientWidth, 1);
    renderer.setSize(nuovo, nuovo, false);
  });
  osservatore.observe(nodo);

  return {
    imposta(forma: string) {
      if (forma === corrente) return;
      precedente = corrente;
      corrente = bersagli.has(forma) ? forma : vicina(forma, bersagli);
      daQuando = performance.now() / 1000;
    },
    smonta() {
      vivo = false;
      osservatore.disconnect();
      renderer.dispose();
      /* Le geometrie e i materiali del modello non li libera il renderer: su
       * una conversazione in cui si cambia personalità più volte, senza
       * questo il browser accumula la memoria video di ogni volto caricato. */
      modello.traverse((oggetto) => {
        const mesh = oggetto as Partial<Mesh>;
        mesh.geometry?.dispose?.();
        const materiale = mesh.material;
        if (Array.isArray(materiale)) materiale.forEach((m) => m.dispose());
        else materiale?.dispose();
      });
      nodo.replaceChildren();
    },
  };
}

interface Riferimento {
  mesh: Mesh;
  indice: number;
}

/** Una mesh che porta davvero i morph target. `three` li dichiara
 *  opzionali su `Mesh` perché la maggior parte delle mesh non ne ha. */
type MeshConPose = Mesh & {
  morphTargetDictionary: Record<string, number>;
  morphTargetInfluences: number[];
};

function haPose(oggetto: Object3D): oggetto is MeshConPose {
  const m = oggetto as Partial<MeshConPose>;
  return !!m.morphTargetDictionary && !!m.morphTargetInfluences;
}

function risolviPose(
  modello: Object3D, pose: Record<string, string>,
): Map<string, Riferimento[]> {
  const bersagli = new Map<string, Riferimento[]>();

  modello.traverse((oggetto) => {
    if (!haPose(oggetto)) return;

    for (const [forma, nome] of Object.entries(pose)) {
      const indice = oggetto.morphTargetDictionary[nome];
      if (indice === undefined) continue;
      const elenco = bersagli.get(forma) ?? [];
      elenco.push({ mesh: oggetto, indice });
      bersagli.set(forma, elenco);
    }
  });

  return bersagli;
}

/* La forma più vicina fra quelle che il modello possiede.
 *
 * Un modello con dieci pose su quindici anima peggio, non male: mandare una
 * forma che non esiste lascerebbe la bocca all'ultima assunta, e su una serie
 * di forme mancanti il volto si bloccherebbe a metà parola. Le sostituzioni
 * sono per luogo di articolazione — le labbra chiuse somigliano più fra loro
 * che a una vocale aperta. */
const VICINE: Record<string, string[]> = {
  A: ["E", "O"],
  E: ["I", "A"],
  I: ["E"],
  O: ["U", "A"],
  U: ["O"],
  MBP: ["FV", "X"],
  FV: ["MBP"],
  L: ["TH", "NN"],
  TH: ["L", "SS"],
  SS: ["TH", "CH"],
  CH: ["SS"],
  KG: ["A", "O"],
  NN: ["L", "TH"],
  RR: ["E", "A"],
  X: [],
};

function vicina(forma: string, bersagli: Map<string, unknown>): string {
  for (const alternativa of VICINE[forma] ?? []) {
    if (bersagli.has(alternativa)) return alternativa;
  }
  return bersagli.has("X") ? "X" : forma;
}

/* Inquadra il volto.
 *
 * Dalla scatola che contiene il modello e non da una posizione fissa: i `.glb`
 * arrivano in scale e origini diverse — chi esporta da Blender in metri, chi
 * in centimetri, chi con i piedi all'origine e chi col centro — e una camera
 * cablata funzionerebbe con un file solo. `camera` nel descrittore permette di
 * correggere a mano quando l'inquadratura automatica sbaglia. */
function inquadra(
  THREE: typeof import("three"),
  modello: Object3D,
  camera: PerspectiveCamera,
  volto: Volto,
): void {
  const camaConfig = (volto.extra?.camera ?? {}) as Record<string, number>;

  const scatola = new THREE.Box3().setFromObject(modello);
  const centro = scatola.getCenter(new THREE.Vector3());
  const misura = scatola.getSize(new THREE.Vector3());

  /* Sul terzo superiore: di una testa interessa il volto, e centrare la
   * scatola metterebbe in mezzo al riquadro il collo. */
  const bersaglio = new THREE.Vector3(
    centro.x,
    camaConfig.y ?? centro.y + misura.y * 0.22,
    centro.z,
  );

  const raggio = Math.max(misura.x, misura.y, misura.z) * 0.5 || 1;
  const distanza =
    camaConfig.distanza ??
    (raggio / Math.tan((camera.fov * Math.PI) / 360)) * 0.9;

  camera.position.set(bersaglio.x, bersaglio.y, bersaglio.z + distanza);
  camera.lookAt(bersaglio);
  camera.updateProjectionMatrix();
}
