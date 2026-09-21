/* Il client della console.
 *
 * Tutte le rotte stanno sotto `/admin` e pretendono il ruolo: un 403 qui non
 * è un guasto da nascondere ma un'informazione — significa che il token
 * dell'utente non porta `admin`, e va detto invece di mostrare una pagina
 * vuota che sembra un errore di caricamento.
 */

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ErroreApi extends Error {
  constructor(
    message: string,
    readonly stato: number,
    readonly dettaglio?: string,
  ) {
    super(message);
  }
}

async function chiamata<T>(
  token: string,
  percorso: string,
  opzioni: RequestInit = {},
): Promise<T> {
  const risposta = await fetch(`${API}${percorso}`, {
    ...opzioni,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...(opzioni.headers ?? {}),
    },
  });

  if (!risposta.ok) {
    let dettaglio: string | undefined;
    try {
      dettaglio = (await risposta.json()).detail;
    } catch {
      /* il corpo non era JSON: il codice di stato basta */
    }
    throw new ErroreApi(
      risposta.status === 403
        ? "Il tuo account non ha il ruolo di amministratore."
        : risposta.status === 401
          ? "La sessione è scaduta."
          : (dettaglio ?? "Richiesta non riuscita."),
      risposta.status,
      dettaglio,
    );
  }

  if (risposta.status === 204) return undefined as T;
  return risposta.json() as Promise<T>;
}

/* ---- forme ---- */

export interface Tipo {
  id: number;
  slug: string;
  name: string;
  description?: string | null;
}

export interface Categoria {
  id: number;
  slug: string;
  name: string;
  rank: number;
}

export interface Versione {
  id: string;
  version: number;
  published_at: string | null;
  corrente: boolean;
  system_prompt: string;
  behavior_rules: Record<string, unknown> | null;
  llm_config: Record<string, unknown> | null;
  rag_config: Record<string, unknown> | null;
  guard_config: Record<string, unknown> | null;
}

export interface CorpusCollegato {
  kb_id: string;
  role: "voice" | "knowledge";
  max_chunks: number;
  enabled: boolean;
}

export interface Personalita {
  id: string;
  slug: string;
  display_name: string;
  description: string | null;
  status: "draft" | "published" | "archived";
  current_version_id: string | null;
  tipi: Tipo[];
  categoria: { id: number; name: string } | null;
}

export interface PersonalitaDettaglio extends Personalita {
  versioni: Versione[];
  corpora: CorpusCollegato[];
}

export interface Base {
  id: string;
  slug: string;
  name: string;
  description: string | null;
  kind: string;
  visibility: string;
  embed_model: string;
  text_config: string;
  stats: {
    documenti?: number;
    passaggi?: number;
    token_stimati?: number;
  } | null;
}

export interface Documento {
  id: string;
  title: string;
  uri: string | null;
  sha256: string;
  meta: Record<string, unknown> | null;
  fetched_at: string | null;
}

export interface PassaggioProvato {
  etichetta: string | null;
  documento: string;
  sezione: string | null;
  testo: string;
  rrf: number;
  posizione_vettoriale: number | null;
  posizione_lessicale: number | null;
  somiglianza: number | null;
  rilevanza_lessicale: number | null;
  trovato_da_entrambe: boolean;
}

export interface EsitoProva {
  domanda: string;
  scelti: PassaggioProvato[];
  scartati: PassaggioProvato[];
}

export interface Opzione {
  label: string;
  available: boolean;
  reason: string;
}

export interface OpzioniRealizzazione {
  modi: Record<string, Opzione>;
  esportazioni: Record<string, Opzione>;
  contesto: Record<string, Opzione>;
}

export interface VoceRegistro {
  id: number;
  action: string;
  actor_id: number | null;
  target_type: string | null;
  target_id: string | null;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  ip: string | null;
  correlation_id: string | null;
  created_at: string;
}

/* ---- personalità ---- */

export const elencoPersonalita = (t: string) =>
  chiamata<Personalita[]>(t, "/admin/personalities");

export const dettaglioPersonalita = (t: string, id: string) =>
  chiamata<PersonalitaDettaglio>(t, `/admin/personalities/${id}`);

export const creaPersonalita = (
  t: string,
  corpo: { slug: string; display_name: string; description?: string },
) =>
  chiamata<Personalita>(t, "/admin/personalities", {
    method: "POST",
    body: JSON.stringify(corpo),
  });

export const modificaPersonalita = (
  t: string,
  id: string,
  corpo: Record<string, unknown>,
) =>
  chiamata<Personalita>(t, `/admin/personalities/${id}`, {
    method: "PATCH",
    body: JSON.stringify(corpo),
  });

export const creaVersione = (
  t: string,
  id: string,
  corpo: Record<string, unknown>,
) =>
  chiamata<{ id: string; version: number }>(
    t,
    `/admin/personalities/${id}/versions`,
    { method: "POST", body: JSON.stringify(corpo) },
  );

export const pubblicaVersione = (t: string, id: string, versionId: string) =>
  chiamata<Personalita>(
    t,
    `/admin/personalities/${id}/versions/${versionId}/publish`,
    { method: "POST" },
  );

export const archiviaPersonalita = (t: string, id: string) =>
  chiamata<Personalita>(t, `/admin/personalities/${id}/archive`, {
    method: "POST",
  });

export const impostaTipi = (t: string, id: string, typeIds: number[]) =>
  chiamata<Personalita>(t, `/admin/personalities/${id}/types`, {
    method: "PUT",
    body: JSON.stringify(typeIds),
  });

export const collegaCorpus = (
  t: string,
  id: string,
  corpo: { kb_id: string; role: string; max_chunks: number },
) =>
  chiamata<{ collegato: boolean }>(t, `/admin/personalities/${id}/corpora`, {
    method: "POST",
    body: JSON.stringify(corpo),
  });

export const scollegaCorpus = (t: string, id: string, kbId: string) =>
  chiamata<{ scollegato: boolean }>(
    t,
    `/admin/personalities/${id}/corpora/${kbId}`,
    { method: "DELETE" },
  );

/* ---- conoscenza ---- */

export const elencoBasi = (t: string) =>
  chiamata<Base[]>(t, "/admin/knowledge-bases");

export const creaBase = (
  t: string,
  corpo: { slug: string; name: string; description?: string; kind?: string },
) =>
  chiamata<{ id: string; slug: string; embed_model: string }>(
    t,
    "/admin/knowledge-bases",
    { method: "POST", body: JSON.stringify(corpo) },
  );

export const documentiDi = (t: string, kbId: string) =>
  chiamata<Documento[]>(t, `/admin/knowledge-bases/${kbId}/documents`);

export const eliminaDocumento = (t: string, kbId: string, docId: string) =>
  chiamata<{ eliminato: boolean }>(
    t,
    `/admin/knowledge-bases/${kbId}/documents/${docId}`,
    { method: "DELETE" },
  );

export const provaRecupero = (
  t: string,
  corpo: { domanda: string; kb_ids: string[]; limite: number },
) =>
  chiamata<EsitoProva>(t, "/admin/retrieval/preview", {
    method: "POST",
    body: JSON.stringify(corpo),
  });

/* ---- tassonomia e stato ---- */

export const elencoTipi = (t: string) => chiamata<Tipo[]>(t, "/admin/types");

export const creaTipo = (t: string, corpo: { slug: string; name: string }) =>
  chiamata<Tipo>(t, "/admin/types", {
    method: "POST",
    body: JSON.stringify(corpo),
  });

export const elencoCategorie = (t: string) =>
  chiamata<Categoria[]>(t, "/admin/categories");

export const opzioniRealizzazione = (t: string) =>
  chiamata<OpzioniRealizzazione>(t, "/admin/build-options");

export const registro = (t: string, filtro?: string) =>
  chiamata<VoceRegistro[]>(
    t,
    `/admin/audit${filtro ? `?azione=${encodeURIComponent(filtro)}` : ""}`,
  );
