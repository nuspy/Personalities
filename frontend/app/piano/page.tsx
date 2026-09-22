"use client";

/* Il piano: cosa si ha, quanto resta, cosa si può scegliere.
 *
 * **Il ritorno dal pagamento non si fida dell'indirizzo.** Chi torna con
 * `?checkout=…` potrebbe averlo scritto a mano: lo stato vero si chiede al
 * server, che lo cambia solo quando arriva l'evento firmato del fornitore.
 */

import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import {
  abbonaGratis,
  apriPagamento,
  disdici,
  elencaPiani,
  leggiConto,
  leggiMovimenti,
  statoPagamento,
  type Conto,
  type Piano,
} from "@/lib/api";
import { useAzione, useCarica, useToken } from "@/lib/usa";
import { Involucro } from "../navigazione";
import stili from "../pagine.module.css";

const NOMI_LIMITI: Record<string, string> = {
  messaggi_al_giorno: "Messaggi nelle ultime 24 ore",
  conversazioni: "Conversazioni aperte",
  memorie: "Memorie",
};

const MOTIVI: Record<string, string> = {
  accredito_piano: "Crediti del piano",
  consumo: "Risposta",
  rimborso: "Rimborso",
  rettifica: "Correzione",
};

function euro(centesimi: number): string {
  return new Intl.NumberFormat("it-IT", { style: "currency", currency: "EUR" }).format(centesimi / 100);
}

export default function PaginaPiano() {
  return (
    <Involucro
      titolo="Il tuo piano"
      sottotitolo="Il piano decide con quali personalità puoi parlare e quanto; i crediti sono ciò che ogni risposta consuma."
    >
      {/* `useSearchParams` pretende un confine di sospensione in Next 16:
          senza, la pagina non si potrebbe generare in anticipo. */}
      <Suspense fallback={null}>
        <Contenuto />
      </Suspense>
    </Involucro>
  );
}

function Contenuto() {
  const conto = useCarica(leggiConto);
  const piani = useCarica(useCallback(() => elencaPiani(), []));
  const movimenti = useCarica(leggiMovimenti);
  const ritorno = useRitornoDalPagamento(conto.ricarica);

  const ricaricaTutto = () => {
    conto.ricarica();
    movimenti.ricarica();
  };

  return (
    <>
      {ritorno && (
        <p className={ritorno.buono ? stili.avvisoBuono : stili.avviso} role="status">
          {ritorno.messaggio}
        </p>
      )}

      {conto.errore && <p className={stili.errore}>{conto.errore}</p>}
      {conto.dati && <Attuale conto={conto.dati} suCambio={ricaricaTutto} />}

      <section className={stili.sezione}>
        <h2 className={stili.sezioneTitolo}>Piani</h2>
        {piani.errore && <p className={stili.errore}>{piani.errore}</p>}
        {piani.dati && conto.dati && (
          <Catalogo piani={piani.dati} conto={conto.dati} suCambio={ricaricaTutto} />
        )}
      </section>

      <section className={stili.sezione}>
        <h2 className={stili.sezioneTitolo}>Movimenti recenti</h2>
        {movimenti.dati && movimenti.dati.movimenti.length === 0 && (
          <div className={stili.vuoto}>Nessun movimento ancora.</div>
        )}
        {movimenti.dati && movimenti.dati.movimenti.length > 0 && (
          <ul className={stili.elenco}>
            {movimenti.dati.movimenti.map((m, i) => (
              <li key={i} className={stili.movimento}>
                <div>
                  <p className={stili.rigaTesto}>{MOTIVI[m.reason] ?? m.reason}</p>
                  <p className={stili.rigaDettagli}>
                    {new Date(m.quando).toLocaleString("it-IT")}
                    {m.note ? ` · ${m.note}` : ""}
                  </p>
                </div>
                <span className={m.delta >= 0 ? stili.positivo : stili.negativo}>
                  {m.delta > 0 ? "+" : ""}{m.delta}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

/* ---- il piano attuale ---- */

function Attuale({ conto, suCambio }: { conto: Conto; suCambio: () => void }) {
  const { esegui, inCorso, errore } = useAzione();
  const [conferma, setConferma] = useState(false);
  const abbonamento = conto.abbonamento;
  const disdetto = !!abbonamento?.disdetto_il;

  return (
    <section className={stili.sezione}>
      <div className={stili.attuale}>
        <div>
          <div className={stili.attualeNome}>{abbonamento?.nome ?? "Gratuito"}</div>
          <p className={stili.nota}>
            {abbonamento
              ? disdetto
                ? `Disdetto: resta attivo fino al ${new Date(abbonamento.periodo_fine).toLocaleDateString("it-IT")}.`
                : `Si rinnova il ${new Date(abbonamento.periodo_fine).toLocaleDateString("it-IT")}.`
              : "Accesso gratuito."}
          </p>
        </div>
        <div className={stili.saldo}>
          <div className={stili.saldoNumero}>{conto.saldo}</div>
          <p className={stili.nota}>{conto.saldo === 1 ? "credito" : "crediti"}</p>
        </div>

        {conto.limiti_attivi && conto.uso && (
          <div className={stili.misure}>
            {Object.entries(conto.uso).map(([nome, u]) => {
              const quota = u.limite ? Math.min(1, u.usati / u.limite) : 0;
              return (
                <div key={nome} className={stili.misura}>
                  {NOMI_LIMITI[nome] ?? nome}
                  <div className={stili.traccia} aria-hidden="true">
                    <span
                      className={quota >= 1 ? stili.riempimentoPieno : stili.riempimento}
                      style={{ width: `${quota * 100}%` }}
                    />
                  </div>
                  {u.limite === null ? `${u.usati} · illimitato` : `${u.usati} di ${u.limite}`}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {abbonamento && !disdetto && abbonamento.piano !== "free" && (
        <div style={{ marginTop: "0.75rem" }}>
          {conferma ? (
            <>
              <p className={stili.nota}>
                Il piano resta attivo fino alla fine del periodo già pagato. Confermi?
              </p>
              <div className={stili.rigaAzioni} style={{ justifyContent: "flex-start", marginTop: "0.5rem" }}>
                <button
                  className={stili.distruttiva}
                  disabled={inCorso}
                  onClick={async () => {
                    if (await esegui((t) => disdici(t))) suCambio();
                    setConferma(false);
                  }}
                >
                  Sì, disdici
                </button>
                <button className={stili.secondaria} onClick={() => setConferma(false)}>
                  No
                </button>
              </div>
            </>
          ) : (
            <button className={stili.secondaria} onClick={() => setConferma(true)}>
              Disdici il piano
            </button>
          )}
          {errore && <p className={stili.errore}>{errore}</p>}
        </div>
      )}
    </section>
  );
}

/* ---- il catalogo ---- */

function Catalogo({
  piani,
  conto,
  suCambio,
}: {
  piani: Piano[];
  conto: Conto;
  suCambio: () => void;
}) {
  const [annuale, setAnnuale] = useState(false);
  const [versoGratuito, setVersoGratuito] = useState<Piano | null>(null);
  const { esegui, inCorso, errore } = useAzione();
  const attuale = conto.diritti.piano;
  const pianoAttuale = piani.find((p) => p.slug === attuale);
  const pagaOra = !!pianoAttuale && (pianoAttuale.prezzo_mensile > 0 || pianoAttuale.prezzo_annuale > 0);
  const fine = conto.abbonamento
    ? new Date(conto.abbonamento.periodo_fine).toLocaleDateString("it-IT")
    : "";

  const scegli = async (piano: Piano) => {
    const prezzo = annuale ? piano.prezzo_annuale : piano.prezzo_mensile;
    if (prezzo <= 0) {
      /* Da un piano pagato il gratuito arriva a fine periodo: lo si dice
       * prima, perché una disdetta non si ritira da questa pagina. */
      if (pagaOra && !versoGratuito) {
        setVersoGratuito(piano);
        return;
      }
      setVersoGratuito(null);
      if (await esegui((t) => abbonaGratis(t, piano.slug))) suCambio();
      return;
    }
    const sessione = await esegui((t) => apriPagamento(t, piano.slug, annuale));
    /* Si lascia la pagina per quella del fornitore: il piano si attiva solo
     * quando il fornitore conferma il pagamento, non qui. */
    if (sessione) window.location.assign(sessione.url);
  };

  return (
    <>
      <div className={stili.periodo} role="group" aria-label="Periodo di fatturazione">
        <button aria-pressed={!annuale} onClick={() => setAnnuale(false)}>Mensile</button>
        <button aria-pressed={annuale} onClick={() => setAnnuale(true)}>Annuale</button>
      </div>

      <div className={stili.catalogo}>
        {piani.map((p) => {
          const prezzo = annuale ? p.prezzo_annuale : p.prezzo_mensile;
          const eAttuale = p.slug === attuale;
          const limiteMessaggi = p.limiti?.messaggi_al_giorno;
          return (
            <article key={p.slug} className={eAttuale ? stili.pianoAttuale : stili.piano}>
              <div className={stili.pianoNome}>{p.nome}</div>
              <div className={stili.prezzo}>
                {prezzo > 0 ? euro(prezzo) : "Gratis"}
                {prezzo > 0 && <small> / {annuale ? "anno" : "mese"}</small>}
              </div>
              <ul className={stili.voci}>
                <li>{p.crediti_per_periodo} crediti per periodo</li>
                <li>
                  {limiteMessaggi === undefined || limiteMessaggi < 0
                    ? "Messaggi illimitati"
                    : `${limiteMessaggi} messaggi al giorno`}
                </li>
                <li>{p.diritti?.voce ? "Risposte a voce" : "Risposte scritte"}</li>
              </ul>
              {/* La conferma sta nella scheda premuta: sul telefono il fondo
                  del catalogo è tre schermate più in basso, e una domanda che
                  non si vede è un pulsante che sembra non fare nulla. */}
              {versoGratuito?.slug === p.slug ? (
                <div className={stili.conferma} role="alertdialog" aria-label="Conferma il passaggio al gratuito">
                  <p>
                    {pianoAttuale?.nome} resta attivo fino al {fine}, già pagato;
                    da lì passi a {p.nome} e il rinnovo non viene addebitato.
                  </p>
                  <div className={stili.rigaAzioni} style={{ justifyContent: "flex-start" }}>
                    <button className={stili.primaria} disabled={inCorso} onClick={() => scegli(p)} autoFocus>
                      Sì, a fine periodo
                    </button>
                    <button className={stili.secondaria} onClick={() => setVersoGratuito(null)}>
                      No
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  className={eAttuale ? stili.secondaria : stili.primaria}
                  disabled={eAttuale || inCorso}
                  onClick={() => scegli(p)}
                >
                  {eAttuale
                    ? "Il tuo piano"
                    : prezzo > 0
                      ? "Passa a questo piano"
                      : pagaOra
                        ? "Torna al gratuito"
                        : "Scegli"}
                </button>
              )}
            </article>
          );
        })}
      </div>
      {errore && <p className={stili.errore}>{errore}</p>}
    </>
  );
}

/* ---- il ritorno dal pagamento ---- */

function useRitornoDalPagamento(ricarica: () => void) {
  const parametri = useSearchParams();
  const token = useToken();
  const id = parametri.get("checkout");
  const [esito, setEsito] = useState<{ messaggio: string; buono: boolean } | null>(null);

  useEffect(() => {
    if (!id || !token) return;
    let vivo = true;
    let tentativi = 0;

    /* L'evento del fornitore può arrivare un attimo dopo il ritorno
     * dell'utente: si riprova per qualche secondo invece di dichiarare un
     * pagamento non riuscito che invece sta per essere confermato. */
    const controlla = async () => {
      try {
        const stato = await statoPagamento(token, id);
        if (!vivo) return;
        if (stato.stato === "pagato") {
          setEsito({ messaggio: `Pagamento riuscito: il piano ${stato.nome} è attivo.`, buono: true });
          ricarica();
          return;
        }
        if (stato.stato === "annullato") {
          setEsito({ messaggio: "Pagamento annullato: il piano non è cambiato.", buono: false });
          return;
        }
        if (++tentativi < 6) setTimeout(controlla, 1000);
        else setEsito({ messaggio: "Il pagamento non è ancora confermato: ricontrolla fra poco.", buono: false });
      } catch {
        if (vivo) setEsito({ messaggio: "Non riesco a verificare il pagamento.", buono: false });
      }
    };
    controlla();
    return () => { vivo = false; };
  }, [id, token, ricarica]);

  return esito;
}
