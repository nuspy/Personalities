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
}

export interface MessaggioSalvato {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
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
  | { tipo: "verifica"; verifica: Verifica };

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
    throw new ErroreApi(
      risposta.status === 401
        ? "La sessione è scaduta."
        : "Richiesta non riuscita.",
      risposta.status,
    );
  }
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
    default:
      return null;
  }
}

/* ---- conto ---- */

export interface Conto {
  abbonamento: {
    piano: string;
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
