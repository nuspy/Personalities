"use client";

/* I grafici della console: stat tile, colonne nel tempo, barre per categoria.
 *
 * Regole che valgono per tutti, e per cui esistono come componenti invece di
 * essere ridisegnati pagina per pagina:
 *
 * - **una serie, un colore, nessuna legenda**: il titolo dice cosa si vede;
 * - **segni sottili** — colonne e barre al massimo 24 px, estremità
 *   arrotondata di 4 px, 2 px di carta fra un segno e l'altro;
 * - **etichette parche**: il valore sul massimo e sull'ultimo, il resto nel
 *   tooltip e nella tabella;
 * - **il testo non prende il colore della serie**: valori ed etichette usano
 *   l'inchiostro, il colore sta solo sui segni;
 * - **ogni grafico ha la sua tabella**, perché il tooltip aiuta ma non deve
 *   essere l'unica strada per un numero.
 *
 * Il colore della serie è il primo della palette di riferimento, validato
 * su entrambe le superfici della console (`validate_palette.js`: tutti i
 * controlli superati in chiaro e in scuro).
 */

import { useEffect, useState } from "react";
import stili from "./grafici.module.css";

const numeri = new Intl.NumberFormat("it-IT");

export function compatto(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(".", ",")} Mln`;
  if (Math.abs(n) >= 10_000) return `${(n / 1_000).toFixed(1).replace(".", ",")}k`;
  return numeri.format(n);
}

export function percento(q: number | null | undefined, cifre = 0): string {
  return q === null || q === undefined ? "—" : `${(q * 100).toFixed(cifre).replace(".", ",")}%`;
}

/** Tacche pulite: 0, 5, 10… o 0, 200, 400…, mai 0, 37, 74. */
function tacche(massimo: number, quante = 4): number[] {
  if (massimo <= 0) return [0, 1];
  const grezzo = massimo / quante;
  const grandezza = 10 ** Math.floor(Math.log10(grezzo));
  const passo = [1, 2, 5, 10].map((m) => m * grandezza).find((p) => p >= grezzo) ?? grezzo;
  const cima = Math.ceil(massimo / passo) * passo;
  const uscita: number[] = [];
  for (let v = 0; v <= cima + passo / 2; v += passo) uscita.push(v);
  return uscita;
}

/* Un riferimento a funzione e non `useRef`: il contenitore sparisce quando
 * si passa alla tabella e ne nasce un altro al ritorno, e l'osservatore deve
 * seguire quello nuovo — con `useRef` resterebbe attaccato al vecchio, e il
 * grafico smetterebbe di adattarsi alla finestra. */
function useLarghezza<T extends HTMLElement>() {
  const [elemento, setElemento] = useState<T | null>(null);
  const [larghezza, setLarghezza] = useState(0);
  useEffect(() => {
    if (!elemento) return;
    const osservatore = new ResizeObserver(([voce]) => setLarghezza(voce.contentRect.width));
    osservatore.observe(elemento);
    return () => osservatore.disconnect();
  }, [elemento]);
  return [setElemento, larghezza] as const;
}

/* ---- stat tile ---- */

export function Tessera({
  etichetta,
  valore,
  precedente,
  formato = compatto,
  meglio = "su",
  nota,
}: {
  etichetta: string;
  valore: number | null;
  precedente?: number | null;
  formato?: (n: number) => string;
  /** Se salire è una buona notizia: per la latenza e gli errori non lo è. */
  meglio?: "su" | "giu";
  nota?: string;
}) {
  const delta =
    valore !== null && precedente !== null && precedente !== undefined ? valore - precedente : null;
  const direzione = delta === null || delta === 0 ? "pari" : delta > 0 ? "su" : "giu";
  const buona = direzione !== "pari" && direzione === meglio;

  return (
    <div className={stili.tessera}>
      <div className={stili.tesseraEtichetta}>{etichetta}</div>
      <div className={stili.tesseraValore}>{valore === null ? "—" : formato(valore)}</div>
      {delta !== null && (
        <div
          className={
            direzione === "pari" ? stili.deltaPari : buona ? stili.deltaBuono : stili.deltaCattivo
          }
        >
          {/* La freccia dice la direzione, la parola dice se è bene: il colore
              da solo non basta a chi non lo distingue. */}
          <span aria-hidden="true">{direzione === "su" ? "▲" : direzione === "giu" ? "▼" : "="}</span>{" "}
          {direzione === "pari"
            ? "come il periodo prima"
            : `${delta > 0 ? "+" : "−"}${formato(Math.abs(delta))} sul periodo prima`}
        </div>
      )}
      {nota && <div className={stili.tesseraNota}>{nota}</div>}
    </div>
  );
}

/* ---- colonne nel tempo ---- */

export interface Punto {
  chiave: string;
  etichetta: string;
  valore: number;
}

export function Colonne({
  titolo,
  descrizione,
  punti,
  unita,
}: {
  titolo: string;
  descrizione?: string;
  punti: Punto[];
  unita: string;
}) {
  const [contenitore, larghezza] = useLarghezza<HTMLDivElement>();
  const [scelto, setScelto] = useState<number | null>(null);
  const [tabella, setTabella] = useState(false);

  const altezza = 220;
  const margine = { sopra: 18, destra: 8, sotto: 26, sinistra: 40 };
  const utile = Math.max(0, larghezza - margine.sinistra - margine.destra);
  const alto = altezza - margine.sopra - margine.sotto;
  const massimo = Math.max(0, ...punti.map((p) => p.valore));
  const scala = tacche(massimo);
  const cima = scala[scala.length - 1] || 1;
  const banda = punti.length ? utile / punti.length : 0;
  // Sottile: mai più di 24 px, e sempre 2 px di carta fra una e l'altra.
  const spessore = Math.max(1, Math.min(24, banda - 2));
  const y = (v: number) => margine.sopra + alto - (v / cima) * alto;

  const indiceMassimo = punti.reduce((m, p, i) => (p.valore > (punti[m]?.valore ?? -1) ? i : m), 0);
  const ultimo = punti.length - 1;
  // Le etichette sull'asse: poche, a distanza leggibile.
  const ogni = Math.max(1, Math.ceil(punti.length / Math.max(1, Math.floor(utile / 64))));

  const corrente = scelto !== null ? punti[scelto] : null;

  return (
    <figure className={stili.figura}>
      <figcaption className={stili.intestazione}>
        <div>
          <div className={stili.titolo}>{titolo}</div>
          {descrizione && <div className={stili.descrizione}>{descrizione}</div>}
        </div>
        <button className={stili.comandoTabella} onClick={() => setTabella((v) => !v)} aria-pressed={tabella}>
          {tabella ? "Grafico" : "Tabella"}
        </button>
      </figcaption>

      {tabella ? (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr><th>Giorno</th><th className={stili.numero}>{unita}</th></tr>
            </thead>
            <tbody>
              {punti.map((p) => (
                <tr key={p.chiave}><td>{p.etichetta}</td><td className={stili.numero}>{numeri.format(p.valore)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div ref={contenitore} className={stili.area}>
          {larghezza > 0 && (
            <svg
              width={larghezza}
              height={altezza}
              className={stili.svg}
              tabIndex={0}
              role="img"
              aria-label={`${titolo}: ${punti.length} giorni, massimo ${numeri.format(massimo)} ${unita}. Frecce per scorrere i giorni.`}
              onPointerLeave={() => setScelto(null)}
              onBlur={() => setScelto(null)}
              onFocus={() => setScelto((s) => s ?? ultimo)}
              onKeyDown={(e) => {
                if (e.key === "ArrowLeft") setScelto((s) => Math.max(0, (s ?? ultimo) - 1));
                if (e.key === "ArrowRight") setScelto((s) => Math.min(ultimo, (s ?? ultimo) + 1));
              }}
            >
              {scala.map((v) => (
                <g key={v}>
                  <line x1={margine.sinistra} x2={larghezza - margine.destra} y1={y(v)} y2={y(v)} className={stili.griglia} />
                  <text x={margine.sinistra - 6} y={y(v)} className={stili.tacca} textAnchor="end" dominantBaseline="middle">
                    {compatto(v)}
                  </text>
                </g>
              ))}

              {punti.map((p, i) => {
                const centro = margine.sinistra + banda * i + banda / 2;
                const cimaColonna = y(p.valore);
                const h = margine.sopra + alto - cimaColonna;
                const raggio = Math.min(4, spessore / 2, h);
                const x0 = centro - spessore / 2;
                const base = margine.sopra + alto;
                // Estremità arrotondata in cima, squadrata sulla base.
                const percorso = h <= 0 ? "" : `M${x0},${base} V${cimaColonna + raggio} Q${x0},${cimaColonna} ${x0 + raggio},${cimaColonna} H${x0 + spessore - raggio} Q${x0 + spessore},${cimaColonna} ${x0 + spessore},${cimaColonna + raggio} V${base} Z`;
                const etichettata = p.valore > 0 && (i === indiceMassimo || i === ultimo);
                return (
                  <g key={p.chiave}>
                    {percorso && (
                      <path d={percorso} className={scelto === i ? stili.colonnaAccesa : stili.colonna} />
                    )}
                    {etichettata && (
                      <text x={centro} y={cimaColonna - 5} textAnchor="middle" className={stili.valore}>
                        {compatto(p.valore)}
                      </text>
                    )}
                    {i % ogni === 0 && (
                      <text x={centro} y={altezza - 8} textAnchor="middle" className={stili.tacca}>
                        {p.etichetta}
                      </text>
                    )}
                    {/* Il bersaglio è tutta la banda, non il segno: una colonna
                        da 3 px non la centra nessuno. */}
                    <rect
                      x={margine.sinistra + banda * i}
                      y={margine.sopra}
                      width={banda}
                      height={alto}
                      fill="transparent"
                      onPointerEnter={() => setScelto(i)}
                      onPointerMove={() => setScelto(i)}
                    />
                  </g>
                );
              })}
              <line
                x1={margine.sinistra} x2={larghezza - margine.destra}
                y1={margine.sopra + alto} y2={margine.sopra + alto}
                className={stili.asse}
              />
            </svg>
          )}
          {corrente && scelto !== null && (
            <div
              className={stili.suggerimento}
              style={{
                left: Math.min(
                  Math.max(margine.sinistra + banda * scelto + banda / 2, 60),
                  Math.max(60, larghezza - 60),
                ),
              }}
              role="status"
            >
              <strong>{numeri.format(corrente.valore)}</strong>
              <span>{unita} · {corrente.etichetta}</span>
            </div>
          )}
        </div>
      )}
    </figure>
  );
}

/* ---- barre per categoria ---- */

export interface Barra {
  chiave: string;
  etichetta: string;
  /** `null` quando il valore non esiste — un'approvazione senza voti — e
   *  va detto «—», non disegnato come uno zero. */
  valore: number | null;
  dettaglio?: string;
}

export function Barre({
  titolo,
  descrizione,
  barre,
  unita,
  colonneTabella,
}: {
  titolo: string;
  descrizione?: string;
  barre: Barra[];
  unita: string;
  /** Le colonne in più della tabella: quello che il grafico non mostra. */
  colonneTabella?: { titolo: string; valori: string[] }[];
}) {
  const [tabella, setTabella] = useState(false);
  const massimo = Math.max(1, ...barre.map((b) => b.valore ?? 0));

  return (
    <figure className={stili.figura}>
      <figcaption className={stili.intestazione}>
        <div>
          <div className={stili.titolo}>{titolo}</div>
          {descrizione && <div className={stili.descrizione}>{descrizione}</div>}
        </div>
        <button className={stili.comandoTabella} onClick={() => setTabella((v) => !v)} aria-pressed={tabella}>
          {tabella ? "Grafico" : "Tabella"}
        </button>
      </figcaption>

      {barre.length === 0 ? (
        <p className={stili.vuoto}>Nessun dato nel periodo.</p>
      ) : tabella ? (
        <div className={stili.contenitoreTabella}>
          <table className={stili.tabella}>
            <thead>
              <tr>
                <th />
                <th className={stili.numero}>{unita}</th>
                {colonneTabella?.map((c) => <th key={c.titolo} className={stili.numero}>{c.titolo}</th>)}
              </tr>
            </thead>
            <tbody>
              {barre.map((b, i) => (
                <tr key={b.chiave}>
                  <td>{b.etichetta}</td>
                  <td className={stili.numero}>{b.valore === null ? "—" : numeri.format(b.valore)}</td>
                  {colonneTabella?.map((c) => <td key={c.titolo} className={stili.numero}>{c.valori[i]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ul className={stili.barre}>
          {barre.map((b) => (
            <li key={b.chiave} className={stili.barraRiga} title={b.dettaglio}>
              <span className={stili.barraEtichetta}>{b.etichetta}</span>
              <span className={stili.barraTraccia}>
                {b.valore !== null && (
                  <span className={stili.barra} style={{ width: `${(b.valore / massimo) * 100}%` }} />
                )}
                <span className={stili.barraValore}>{b.valore === null ? "—" : numeri.format(b.valore)}</span>
              </span>
              {b.dettaglio && <span className={stili.barraDettaglio}>{b.dettaglio}</span>}
            </li>
          ))}
        </ul>
      )}
    </figure>
  );
}
