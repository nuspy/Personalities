/* Il client dell'API.
 *
 * `EventSource` sarebbe la via naturale per SSE, ma non serve qui: non accetta
 * intestazioni, quindi non può portare il token, e non fa POST. Si legge il
 * corpo della risposta con i `ReadableStream` di `fetch`, che danno anche il
 * controllo per interrompere una generazione a metà.
 */

const API =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
    throw new ErroreApi(
      risposta.status === 401
        ? "La sessione è scaduta."
        : risposta.status === 404
          ? "Questa conversazione non esiste più."
          : "Il servizio non ha risposto.",
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
