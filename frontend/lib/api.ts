/* Il client dell'API.
 *
 * `EventSource` sarebbe la via naturale per SSE, ma non serve qui: non accetta
 * intestazioni, quindi non può portare il token, e non fa POST. Si legge il
 * corpo della risposta con i `ReadableStream` di `fetch`, che danno anche il
 * controllo per interrompere una generazione a metà.
 */

const API =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";

export interface Conversazione {
  id: string;
  title: string | null;
  last_message_at: string | null;
  personality: string | null;
}

export interface MessaggioSalvato {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  voto?: number | null;
}

export interface Consumo {
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  total_tokens: number;
  model: string;
}

export interface Fonte {
  etichetta: string;
  documento: string;
  sezione: string | null;
  uri: string | null;
  estratto: string;
}

export interface Personalita {
  slug: string;
  display_name: string;
  description: string | null;
}

export interface Verifica {
  livello: string;
  fondata: boolean;
  non_eseguito: string;
  infondate: { testo: string; nota: string }[];
  conteggi: { fatti: number; infondate: number; totale: number };
}

export type EventoChat =
  | { tipo: "inizio"; conversationId: string; correlationId: string; personalita: string | null }
  | { tipo: "fonti"; fonti: Fonte[] }
  | { tipo: "token"; testo: string }
  | { tipo: "pensa" }
  | { tipo: "degradato"; motivo: string }
  | { tipo: "errore"; messaggio: string; recuperabile: boolean }
  | { tipo: "fine"; consumo: Consumo | null; inventati: string[] }
  /* Arriva **dopo** `fine`: il giudizio costa una chiamata intera, e chi
     ha già finito di leggere non deve aspettarlo. */
  | { tipo: "verifica"; verifica: Verifica }
  /* Per ultimo: la risposta salvata, con l'identificativo per votarla. */
  | { tipo: "salvato"; messageId: string };

export class ErroreApi extends Error {
  constructor(message: string, readonly stato: number) {
    super(message);
  }
}

function intestazioni(token: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

async function leggi<T>(risposta: Response): Promise<T> {
  if (!risposta.ok) {
    /* Il motivo del server, quando c'è: per 402, 403, 409 e 429 è proprio
     * ciò che serve all'utente — «servono 3 crediti», «il piano non
     * comprende questa voce», «archivia una conversazione». Un messaggio
     * generico lo trasformerebbe in un guasto apparente. */
    let motivo: string | undefined;
    try {
      const corpo = await risposta.json();
      motivo = typeof corpo?.detail === "string" ? corpo.detail : undefined;
    } catch {
      /* il corpo non era JSON: restano i codici */
    }
    throw new ErroreApi(
      risposta.status === 401
        ? "La sessione è scaduta."
        : (motivo ?? "Richiesta non riuscita."),
      risposta.status,
    );
  }
  if (risposta.status === 204) return undefined as T;
  return risposta.json() as Promise<T>;
}

export async function elencaPersonalita(): Promise<Personalita[]> {
  /* Senza token: il catalogo è ciò che si vede prima di entrare. */
  return leggi(await fetch(`${API}/personalities`));
}

export async function elencaConversazioni(token: string): Promise<Conversazione[]> {
  return leggi(
    await fetch(`${API}/conversations`, { headers: intestazioni(token) }),
  );
}

export async function leggiMessaggi(
  token: string,
  conversationId: string,
): Promise<MessaggioSalvato[]> {
  return leggi(
    await fetch(`${API}/conversations/${conversationId}/messages`, {
      headers: intestazioni(token),
    }),
  );
}

export interface CapacitaPiattaforma {
  features: Record<
    string,
    { available: boolean; label: string; reason: string }
  >;
  workers: { worker_id: string; role: string; gpu: string; vram_mb: number }[];
}

export async function leggiCapacita(): Promise<CapacitaPiattaforma> {
  return leggi(await fetch(`${API}/capabilities`));
}

/** Manda un messaggio e restituisce gli eventi via via che arrivano. */
export async function* conversa(
  token: string,
  messaggio: string,
  conversationId: string | null,
  personalita: string | null,
  segnale?: AbortSignal,
): AsyncGenerator<EventoChat> {
  const risposta = await fetch(`${API}/chat`, {
    method: "POST",
    headers: intestazioni(token),
    body: JSON.stringify({
      message: messaggio,
      conversation_id: conversationId,
      personality: personalita,
    }),
    signal: segnale,
  });

  if (!risposta.ok || !risposta.body) {
    /* Il motivo lo manda il servizio, e per 402 e 403 è l'unica cosa utile:
     * «servono 3 crediti, ne hai 0» e «il tuo piano non comprende questa
     * voce» si risolvono in modi diversi, e tradurli entrambi in «il
     * servizio non ha risposto» trasforma un limite previsto in un guasto
     * apparente — con l'utente che ricarica invece di abbonarsi. */
    let motivo: string | undefined;
    try {
      motivo = (await risposta.json())?.detail;
    } catch {
      /* il corpo non era JSON: restano i codici */
    }

    throw new ErroreApi(
      risposta.status === 401
        ? "La sessione è scaduta."
        : risposta.status === 404
          ? "Questa conversazione non esiste più."
          : (motivo ?? "Il servizio non ha risposto."),
      risposta.status,
    );
  }

  const lettore = risposta.body.getReader();
  const decodificatore = new TextDecoder();
  let avanzo = "";

  while (true) {
    const { done, value } = await lettore.read();
    if (done) break;

    /* `stream: true` perché un carattere multibyte può essere spezzato fra due
     * pacchetti: senza, una lettera accentata a cavallo del confine diventa
     * il carattere di sostituzione. Con l'italiano succede spesso. */
    avanzo += decodificatore.decode(value, { stream: true });

    const blocchi = avanzo.split("\n\n");
    // L'ultimo pezzo può essere un evento incompleto: resta in attesa.
    avanzo = blocchi.pop() ?? "";

    for (const blocco of blocchi) {
      const evento = interpreta(blocco);
      if (evento) yield evento;
    }
  }
}

function interpreta(blocco: string): EventoChat | null {
  let nome = "";
  let dati = "";
  for (const riga of blocco.split("\n")) {
    if (riga.startsWith("event:")) nome = riga.slice(6).trim();
    else if (riga.startsWith("data:")) dati += riga.slice(5).trim();
  }
  if (!nome || !dati) return null;

  let corpo: Record<string, unknown>;
  try {
    corpo = JSON.parse(dati);
  } catch {
    return null;
  }

  switch (nome) {
    case "start":
      return {
        tipo: "inizio",
        conversationId: String(corpo.conversation_id),
        correlationId: String(corpo.correlation_id),
        personalita: (corpo.personality as string) ?? null,
      };
    case "sources":
      return { tipo: "fonti", fonti: (corpo.passaggi as Fonte[]) ?? [] };
    case "degradato":
      return { tipo: "degradato", motivo: String(corpo.motivo) };
    case "token":
      return { tipo: "token", testo: String(corpo.text) };
    case "thinking":
      return { tipo: "pensa" };
    case "error":
      return {
        tipo: "errore",
        messaggio: String(corpo.message),
        recuperabile: Boolean(corpo.recoverable),
      };
    case "done":
      return {
        tipo: "fine",
        consumo: (corpo.usage as Consumo) ?? null,
        inventati: (corpo.riferimenti_inventati as string[]) ?? [],
      };
    case "verifica":
      return { tipo: "verifica", verifica: corpo as unknown as Verifica };
    case "salvato":
      return { tipo: "salvato", messageId: String(corpo.message_id) };
    default:
      return null;
  }
}

/* ---- conto ---- */

export interface Conto {
  abbonamento: {
    piano: string;
    nome: string;
    stato: string;
    periodo_fine: string;
    disdetto_il: string | null;
  } | null;
  diritti: {
    piano: string;
    categorie: string[];
    voce: boolean;
    predefiniti: boolean;
  };
  saldo: number;
  uso?: Record<string, { usati: number; limite: number | null; restanti: number | null }>;
  limiti_attivi?: boolean;
}

export async function leggiConto(token: string): Promise<Conto> {
  return leggi(
    await fetch(`${API}/me/billing`, { headers: intestazioni(token) }),
  );
}

/* ---- voce e volto ---- */

export interface Viseme {
  forma: string;
  inizio: number;
  fine: number;
}

export interface ParolaDetta {
  testo: string;
  inizio: number;
  fine: number;
}

export interface Voce {
  audio: string;
  media_type: string;
  durata: number;
  /** `allineamento`, `fornitore`, `stima`, o vuoto quando i tempi non ci sono.
   *  Vuoto significa che non si deve animare niente: una bocca mossa su tempi
   *  inventati si vede fuori sincrono, ed è peggio di una ferma. */
  origine_tempi: string;
  parole: ParolaDetta[];
  visemi: Viseme[];
}

export interface Volto {
  id: string;
  slug: string;
  nome: string;
  tipo: "immagine" | "video" | "modello";
  uri: string;
  labiale: boolean;
  pose: Record<string, string>;
  extra: Record<string, unknown>;
}

export async function leggiVoce(
  token: string,
  testo: string,
  personalita: string | null,
  labiale: boolean,
): Promise<Voce> {
  return leggi(
    await fetch(`${API}/voice/speak`, {
      method: "POST",
      headers: intestazioni(token),
      body: JSON.stringify({ testo, personality: personalita, labiale }),
    }),
  );
}

export async function leggiVolto(slug: string): Promise<Volto | null> {
  const corpo = await leggi<{ avatar: Volto | null }>(
    await fetch(`${API}/personalities/${slug}/avatar`),
  );
  return corpo.avatar;
}

/** Da base64 a qualcosa che un `<audio>` sa riprodurre. */
export function sorgenteAudio(voce: Voce): string {
  const binario = atob(voce.audio);
  const byte = new Uint8Array(binario.length);
  for (let i = 0; i < binario.length; i++) byte[i] = binario.charCodeAt(i);
  return URL.createObjectURL(new Blob([byte], { type: voce.media_type }));
}

/* ---- piani e pagamento ---- */

export interface Piano {
  slug: string;
  nome: string;
  prezzo_mensile: number;
  prezzo_annuale: number;
  crediti_per_periodo: number;
  limiti: Record<string, number>;
  diritti: { categorie?: string[]; voce?: boolean };
}

export interface Movimento {
  delta: number;
  reason: string;
  quando: string;
  note: string | null;
}

export async function elencaPiani(): Promise<Piano[]> {
  return leggi(await fetch(`${API}/plans`));
}

export async function leggiMovimenti(token: string): Promise<{ saldo: number; movimenti: Movimento[] }> {
  return leggi(await fetch(`${API}/me/credits?limite=20`, { headers: intestazioni(token) }));
}

/** Il piano gratuito si attiva direttamente; gli altri passano dal pagamento.
 *  Da un piano pagato, il passaggio avviene a fine periodo (`passaggio`). */
export async function abbonaGratis(
  token: string, piano: string,
): Promise<{ passaggio?: { piano: string; dal: string } }> {
  return leggi(await fetch(`${API}/me/subscription`, {
    method: "POST", headers: intestazioni(token), body: JSON.stringify({ piano }),
  }));
}

export async function apriPagamento(
  token: string, piano: string, annuale: boolean,
): Promise<{ checkout_id: string; url: string; importo: number; valuta: string }> {
  return leggi(await fetch(`${API}/me/checkout`, {
    method: "POST", headers: intestazioni(token), body: JSON.stringify({ piano, annuale }),
  }));
}

export async function statoPagamento(
  token: string, id: string,
): Promise<{ checkout_id: string; stato: string; piano: string; nome: string }> {
  return leggi(await fetch(`${API}/me/checkout/${id}`, { headers: intestazioni(token) }));
}

export async function disdici(token: string): Promise<unknown> {
  return leggi(await fetch(`${API}/me/subscription/cancel`, {
    method: "POST", headers: intestazioni(token),
  }));
}

/* ---- memorie ---- */

export interface Memoria {
  id: string;
  kind: "identita" | "preferenza" | "fatto" | "impegno" | "sessione";
  content: string;
  importance: number;
  confidence: number;
  times_referenced: number;
  first_seen_at: string;
  last_referenced_at: string | null;
  valid_to: string | null;
  superseded_by: string | null;
  expires_at: string | null;
  viva: boolean;
  personality_id: string | null;
}

export async function elencaMemorie(
  token: string, includiSuperate = false,
): Promise<{ memorie: Memoria[]; conteggi: Record<string, number> }> {
  return leggi(await fetch(
    `${API}/memory${includiSuperate ? "?includi_superate=true" : ""}`,
    { headers: intestazioni(token) },
  ));
}

export async function storiaMemoria(token: string, id: string): Promise<Memoria[]> {
  const corpo = await leggi<{ storia?: Memoria[] } | Memoria[]>(
    await fetch(`${API}/memory/${id}/history`, { headers: intestazioni(token) }),
  );
  return Array.isArray(corpo) ? corpo : (corpo.storia ?? []);
}

export async function aggiungiMemoria(
  token: string, content: string, kind: string,
): Promise<Memoria> {
  return leggi(await fetch(`${API}/memory`, {
    method: "POST", headers: intestazioni(token),
    body: JSON.stringify({ content, kind, importance: 0.7 }),
  }));
}

export async function dimenticaMemoria(token: string, id: string): Promise<unknown> {
  return leggi(await fetch(`${API}/memory/${id}`, {
    method: "DELETE", headers: intestazioni(token),
  }));
}

export async function dimenticaTutto(token: string): Promise<unknown> {
  return leggi(await fetch(`${API}/memory?conferma=true`, {
    method: "DELETE", headers: intestazioni(token),
  }));
}

export async function accessiMemorie(token: string): Promise<
  { action: string; actor_id: number | null; count: number; reason: string | null; created_at: string }[]
> {
  return leggi(await fetch(`${API}/memory/accesses`, { headers: intestazioni(token) }));
}

/* ---- voti sulle risposte ---- */

export async function votaRisposta(
  token: string, messageId: string, voto: 1 | -1, motivo?: string,
): Promise<{ voto: number | null }> {
  return leggi(await fetch(`${API}/messages/${messageId}/feedback`, {
    method: "PUT", headers: intestazioni(token),
    body: JSON.stringify({ voto, motivo: motivo ?? null }),
  }));
}

export async function togliVoto(token: string, messageId: string): Promise<{ voto: null }> {
  return leggi(await fetch(`${API}/messages/${messageId}/feedback`, {
    method: "DELETE", headers: intestazioni(token),
  }));
}

/* ---- conversazioni ---- */

export async function archiviaConversazione(token: string, id: string): Promise<unknown> {
  return leggi(await fetch(`${API}/conversations/${id}/archive`, {
    method: "POST", headers: intestazioni(token),
  }));
}

/* ---- dettatura, ripiego sul server ---- */

export async function trascriviRegistrazione(
  token: string, audio: Blob,
): Promise<{ testo: string; fiducia: number; da_confermare: boolean }> {
  const modulo = new FormData();
  /* L'estensione segue il formato: Safari registra mp4, gli altri webm, e un
   * nome che mente sul contenuto confonde chi decodifica. */
  const estensione = audio.type.includes("mp4") ? "m4a" : audio.type.includes("ogg") ? "ogg" : "webm";
  modulo.append("file", audio, `dettatura.${estensione}`);
  /* Niente `Content-Type` a mano: il browser deve scrivere il confine del
   * multipart, e un'intestazione impostata qui lo cancellerebbe. */
  return leggi(await fetch(`${API}/voice/listen`, {
    method: "POST", headers: { Authorization: `Bearer ${token}` }, body: modulo,
  }));
}
