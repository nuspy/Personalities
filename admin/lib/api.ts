/* Il client della console.
 *
 * Tutte le rotte stanno sotto `/admin` e pretendono il ruolo: un 403 qui non
 * è un guasto da nascondere ma un'informazione — significa che il token
 * dell'utente non porta `admin`, e va detto invece di mostrare una pagina
 * vuota che sembra un errore di caricamento.
 */

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";

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
  return leggi<T>(
    await fetch(`${API}${percorso}`, {
      ...opzioni,
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        ...(opzioni.headers ?? {}),
      },
    }),
  );
}

async function leggi<T>(risposta: Response): Promise<T> {
  if (!risposta.ok) {
    let dettaglio: string | undefined;
    try {
      dettaglio = (await risposta.json()).detail;
    } catch {
      /* il corpo non era JSON: il codice di stato basta */
    }
    throw new ErroreApi(
      /* Il 403 generico è il ruolo che manca; quando il servizio spiega un
       * rifiuto più preciso, vale la sua spiegazione. */
      risposta.status === 403 && !dettaglio
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
  visibility: "pubblica" | "privata";
  current_version_id: string | null;
  avatar_id: string | null;
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

export interface StatoDigestione {
  kb: { id: string; slug: string; name: string };
  passaggi: {
    totale: number;
    etichettati: number;
    da_fare: number;
    scartati: number;
    ripuliti: number;
  };
  per_categoria: Record<string, number>;
  per_provenienza: Record<string, number>;
}

export interface PassaggioEtichettato {
  id: string;
  ordinale: number;
  documento: string;
  sezione: string | null;
  testo: string;
  testo_originale: string | null;
  categoria: string | null;
  categorie: Record<string, number>;
  provenienza: string | null;
  qualita: number | null;
  sintesi: string;
  da_togliere: string;
  scartato: boolean;
  motivo_scarto: string | null;
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

/* ---- caricamento di documenti ---- */

export interface FormatiCaricabili {
  formati: Record<string, string>;
  byte_massimi: number;
  file_massimi: number;
}

export interface EsitoIngestione {
  documenti: number;
  passaggi: number;
  saltati: { nome: string; motivo: string }[];
  falliti: { nome: string; motivo: string }[];
  avvisi: { nome: string; avviso: string }[];
}

export const formatiCaricabili = (t: string) =>
  chiamata<FormatiCaricabili>(t, "/admin/ingestion/formats");

/** Carica file in un corpus. Restituisce il lavoro che li leggerà. */
export async function caricaDocumenti(
  t: string,
  kbId: string,
  file: File[],
  lingua: string | null,
): Promise<{ build_id: string; stato: string; lingua: string | null }> {
  const modulo = new FormData();
  for (const f of file) modulo.append("file", f, f.name);
  if (lingua) modulo.append("lingua", lingua);
  /* Niente `Content-Type` a mano: il confine del multipart lo scrive il
   * browser, e un'intestazione impostata qui lo cancellerebbe. */
  return leggi(
    await fetch(`${API}/admin/knowledge-bases/${kbId}/uploads`, {
      method: "POST",
      headers: { Authorization: `Bearer ${t}` },
      body: modulo,
    }),
  );
}

export const statoDigestione = (t: string, kbId: string) =>
  chiamata<StatoDigestione>(t, `/admin/knowledge-bases/${kbId}/digestion`);

export const passaggiDi = (
  t: string,
  kbId: string,
  filtri: {
    categoria?: string;
    provenienza?: string;
    solo_scartati?: boolean;
    solo_ripuliti?: boolean;
    limite?: number;
    offset?: number;
  } = {},
) => {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(filtri)) {
    if (v !== undefined && v !== "" && v !== false) q.set(k, String(v));
  }
  return chiamata<{ totale: number; passaggi: PassaggioEtichettato[] }>(
    t,
    `/admin/knowledge-bases/${kbId}/chunks?${q}`,
  );
};

export const avviaDigestione = (
  t: string,
  kbId: string,
  corpo: { chi: string; rifai: boolean },
) =>
  chiamata<{ build_id: string; passaggi_da_fare: number; stato: string }>(
    t,
    `/admin/knowledge-bases/${kbId}/digestion`,
    { method: "POST", body: JSON.stringify(corpo) },
  );

export const statoBuild = (t: string, buildId: string) =>
  chiamata<{
    id: string;
    status: string;
    progress: number;
    message: string | null;
    error: string | null;
  }>(t, `/admin/builds/${buildId}`);

/** Segue l'avanzamento di un lavoro finché non finisce.
 *
 * `fetch` e non `EventSource`: quello non sa mandare l'header
 * `Authorization`, e un token in querystring finisce nei log del proxy.
 *
 * Restituisce la funzione per smettere: chi lascia la pagina deve poter
 * chiudere il flusso, o il browser tiene aperta una connessione per venti
 * minuti a un job che nessuno sta più guardando.
 */
export function seguiLavoro(
  token: string,
  buildId: string,
  su: (evento: string, dato: Record<string, unknown>) => void,
): () => void {
  const freno = new AbortController();

  (async () => {
    const risposta = await fetch(`${API}/admin/builds/${buildId}/events`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: freno.signal,
    });
    if (!risposta.ok || !risposta.body) {
      su("errore", { message: "Non riesco a seguire l'avanzamento." });
      return;
    }

    const lettore = risposta.body.getReader();
    const decoder = new TextDecoder();
    let resto = "";

    while (true) {
      const { done, value } = await lettore.read();
      if (done) break;

      resto += decoder.decode(value, { stream: true });
      // Un blocco SSE finisce con una riga vuota; quello che resta dopo
      // l'ultima è un messaggio a metà, e va tenuto per il giro dopo.
      const blocchi = resto.split("\n\n");
      resto = blocchi.pop() ?? "";

      for (const blocco of blocchi) {
        let evento = "message";
        let dato = "";
        for (const riga of blocco.split("\n")) {
          if (riga.startsWith("event:")) evento = riga.slice(6).trim();
          else if (riga.startsWith("data:")) dato += riga.slice(5).trim();
        }
        if (!dato) continue;
        try {
          su(evento, JSON.parse(dato));
        } catch {
          /* un blocco illeggibile non deve fermare il flusso */
        }
      }
    }
  })().catch((e: unknown) => {
    if ((e as Error)?.name !== "AbortError") {
      su("errore", { message: "Il flusso dell'avanzamento si è interrotto." });
    }
  });

  return () => freno.abort();
}

/* ---- avatar ---- */

export interface Avatar {
  id: string;
  slug: string;
  name: string;
  kind: "immagine" | "video" | "modello";
  description: string | null;
  config: Record<string, unknown>;
  descrittore?: {
    tipo: string;
    uri: string;
    labiale: boolean;
    pose: Record<string, string>;
    extra: Record<string, unknown>;
  };
}

export const elencoAvatar = (t: string) =>
  chiamata<Avatar[]>(t, "/admin/avatars");

export const creaAvatar = (
  t: string,
  corpo: {
    slug: string;
    name: string;
    kind: string;
    config: Record<string, unknown>;
    description?: string;
  },
) =>
  chiamata<Avatar>(t, "/admin/avatars", {
    method: "POST",
    body: JSON.stringify(corpo),
  });

export const modificaAvatar = (
  t: string,
  id: string,
  corpo: Record<string, unknown>,
) =>
  chiamata<Avatar>(t, `/admin/avatars/${id}`, {
    method: "PATCH",
    body: JSON.stringify(corpo),
  });

export const eliminaAvatar = (t: string, id: string) =>
  chiamata<{ eliminato: boolean; personalita_senza_volto: number }>(
    t,
    `/admin/avatars/${id}`,
    { method: "DELETE" },
  );

export const volgiAvatar = (t: string, personalityId: string, avatarId: string | null) =>
  chiamata<{ personality_id: string; avatar_id: string | null }>(
    t,
    `/admin/personalities/${personalityId}/avatar`,
    { method: "PUT", body: JSON.stringify({ avatar_id: avatarId }) },
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

/* ---- laboratorio: analisi, esperimenti, prove ---- */

export interface Indicatori {
  risposte: number;
  utenti_attivi: number;
  conversazioni_nuove: number;
  crediti: number;
  voti: { su: number; giu: number; approvazione: number | null };
  citazioni_inventate: number | null;
  verifica: { verificate: number; fondate: number; quota: number | null };
  primo_token_ms: { p50: number | null; p95: number | null };
  cache: { token_prompt: number; token_da_cache: number; quota: number | null };
}

export interface Analisi {
  periodo: { dal: string; al: string; giorni: number };
  indicatori: Indicatori;
  precedente: Indicatori;
  al_giorno: { giorno: string; risposte: number; utenti: number }[];
  per_personalita: {
    slug: string;
    nome: string;
    risposte: number;
    utenti: number;
    su: number;
    giu: number;
    approvazione: number | null;
  }[];
}

export const leggiAnalisi = (t: string, giorni: number) =>
  chiamata<Analisi>(t, `/admin/analytics?giorni=${giorni}`);

export interface VarianteEsperimento {
  version_id: string;
  peso: number;
  etichetta: string;
  versione?: number;
}

export interface Esperimento {
  id: string;
  personality_id: string;
  nome: string;
  ipotesi: string | null;
  stato: "attivo" | "concluso";
  varianti: VarianteEsperimento[];
  iniziato_il: string;
  concluso_il: string | null;
  vincitore: string | null;
  conclusione: string | null;
  personalita?: { slug: string; nome: string };
}

export interface RisultatoVariante {
  etichetta: string;
  version_id: string;
  peso: number;
  conversazioni: number;
  utenti: number;
  risposte: number;
  su: number;
  giu: number;
  approvazione: number | null;
  primo_token_ms_p50: number | null;
  citazioni_inventate: number | null;
  confronto?: {
    tasso_riferimento: number | null;
    tasso_variante: number | null;
    differenza: number | null;
    z: number | null;
    p: number | null;
    verdetto: string;
  };
}

export const elencoEsperimenti = (t: string) =>
  chiamata<Esperimento[]>(t, "/admin/experiments");

export const dettaglioEsperimento = (t: string, id: string) =>
  chiamata<Esperimento & { risultati: { varianti: RisultatoVariante[]; voti_minimi: number } }>(
    t, `/admin/experiments/${id}`,
  );

export const apriEsperimento = (
  t: string,
  personalitaId: string,
  corpo: { nome: string; ipotesi?: string; varianti: { version_id: string; peso: number }[] },
) =>
  chiamata<Esperimento>(t, `/admin/personalities/${personalitaId}/experiments`, {
    method: "POST",
    body: JSON.stringify(corpo),
  });

export const concludiEsperimento = (
  t: string,
  id: string,
  corpo: { vincitore: string | null; conclusione?: string; pubblica: boolean },
) =>
  chiamata<Esperimento & { pubblicata: boolean }>(t, `/admin/experiments/${id}/conclude`, {
    method: "POST",
    body: JSON.stringify(corpo),
  });

export interface EsitoPlayground {
  risposta: string;
  versione: { id: string; numero: number; modificata: boolean };
  prompt: { role: string; content: string }[];
  punto_di_cache: number | null;
  passaggi: { etichetta: string; documento: string; sezione: string | null; estratto: string }[];
  riferimenti_inventati: string[];
  uso: { prompt_tokens: number; completion_tokens: number; cached_tokens: number; total_tokens: number } | null;
  strategia: { nome: string; modo: string; chiave: string; slot: number | null } | null;
  tempi_ms: Record<string, number>;
}

export const provaPrompt = (
  t: string,
  corpo: {
    personality_id: string;
    version_id?: string;
    system_prompt?: string;
    regole?: string[];
    temperature?: number;
    max_chunks?: number;
    domanda: string;
  },
) =>
  chiamata<EsitoPlayground>(t, "/admin/playground", { method: "POST", body: JSON.stringify(corpo) });
