"""Configurazione del backend, letta dall'ambiente.

In un container la configurazione arriva dall'ambiente, non da un file: i
valori cambiano fra sviluppo, staging e produzione mentre l'immagine resta la
stessa. I default qui sono quelli dello sviluppo con Docker Compose, cosi'
`docker compose up` funziona senza predisporre nulla.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERSONA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- identita' del processo -------------------------------------------
    service_name: str = "persona-api"
    #: "api" | "worker-cpu" | "worker-gpu". Decide cosa il processo annuncia
    #: e quali code serve.
    role: Literal["api", "worker-cpu", "worker-gpu"] = "api"
    environment: Literal["dev", "staging", "prod"] = "dev"

    # --- supporti ----------------------------------------------------------
    #: Il valore predefinito e' quello per un processo avviato **sull'host**,
    #: e punta alla 5433 perche' e' li' che Compose espone il database: la 5432
    #: e' spesso gia' occupata da un PostgreSQL di sistema. I container
    #: ricevono `PERSONA_DATABASE_URL` con `postgres:5432` e non usano questo.
    database_url: str = "postgresql+psycopg://persona:persona@localhost:5433/persona"
    redis_url: str = "redis://localhost:6379/0"

    # --- identita' degli utenti -------------------------------------------
    keycloak_url: str = "http://localhost:8080"
    keycloak_realm: str = "personalities"
    keycloak_client_id: str = "persona-api"
    #: L'indirizzo pubblico di Keycloak, quello scritto nel campo `iss` dei
    #: token. Vuoto vale `keycloak_url`, come in sviluppo; in un cluster
    #: `keycloak_url` e' il servizio interno da cui si leggono le chiavi, e
    #: questo e' il dominio che vede il browser.
    keycloak_issuer_url: str = ""
    #: Altri client del realm i cui token questa API accetta. Il frontend ne fa
    #: parte: il token che riceve e' emesso per `persona-frontend`, e se non
    #: fosse elencato qui ogni richiesta dell'interfaccia verrebbe respinta —
    #: con un 401 che sembrerebbe un problema di login. Resta un elenco chiuso
    #: perche' accettare qualunque destinatario significa accettare i token di
    #: qualunque applicazione che usi lo stesso realm.
    keycloak_accepted_audiences: tuple[str, ...] = ("persona-frontend", "persona-admin")
    #: In sviluppo l'autenticazione puo' essere disattivata per provare gli
    #: endpoint senza avviare Keycloak. Va negata fuori dallo sviluppo, e il
    #: controllo e' in `validate_production()`: un flag simile lasciato acceso
    #: per errore e' fra i modi piu' comuni di esporre un servizio.
    auth_disabled: bool = False

    # --- voce --------------------------------------------------------------
    #: Il fornitore di sintesi. Vuoto o irraggiungibile significa che la
    #: piattaforma non parla: la funzione si dichiara indisponibile, non si
    #: nasconde — un pulsante che sparisce sembra un difetto, uno disabilitato
    #: col motivo accanto e' informazione.
    tts_base_url: str = "http://127.0.0.1:1234/v1"
    tts_api_key: str = "non-serve-in-locale"
    tts_model: str = ""
    tts_voice: str = ""
    #: Con `false` la sintesi usa il fornitore muto: serve a provare il
    #: percorso completo — diritto, generazione, allineamento, visemi — su una
    #: macchina senza modello di voce.
    tts_enabled: bool = False
    #: Misurare i tempi delle parole costa una trascrizione dell'audio appena
    #: prodotto. Vale su una GPU; su un nodo CPU raddoppia l'attesa della
    #: risposta parlata, e li' conviene spegnerlo e rinunciare al labiale.
    tts_align_words: bool = True

    # --- pagamenti ---------------------------------------------------------
    #: Chi incassa: `mock` (il simulatore, per lo sviluppo) o `disattivato`
    #: (nessuno: i piani a pagamento rispondono 503). Il fornitore vero non e'
    #: ancora scelto, e il flusso — sessione, pagina ospitata, evento firmato —
    #: e' lo stesso per tutti. In produzione `mock` impedisce l'avvio.
    billing_provider: str = "mock"
    #: Il segreto con cui il fornitore firma i suoi eventi.
    billing_webhook_secret: str = "segreto-di-sviluppo-da-cambiare"
    #: Fa fallire i rinnovi del simulatore, per provare la sospensione.
    billing_mock_renewal_fails: bool = False
    #: Gli indirizzi pubblici: dove sta l'API (per la pagina di pagamento
    #: simulata) e dove torna l'utente dopo aver pagato.
    api_public_url: str = "http://localhost:8100"
    web_public_url: str = "http://localhost:3000"

    # --- caricamenti dalla console -----------------------------------------
    #: Dove la console deposita i file da ingerire finché il worker non li
    #: legge. In Kubernetes è un volume condiviso fra API e worker CPU
    #: (ReadWriteMany): stanno su pod diversi, e un file scritto sul disco
    #: effimero dell'API il worker non lo vedrebbe mai.
    upload_dir: str = "data/caricamenti"
    #: Il limite per file, in megabyte. Un'ora di parlato compresso sta sotto
    #: i cento; oltre, conviene l'ingestione da riga di comando sul nodo.
    upload_max_mb: int = 200
    #: Il modello Whisper per l'audio e il video caricati dalla console.
    transcription_model: str = "large-v3"

    # --- osservabilita' ----------------------------------------------------
    otlp_endpoint: str = "http://localhost:4318"
    tracing_enabled: bool = True
    log_level: str = "INFO"

    # --- modelli -----------------------------------------------------------
    #: Nome dell'applicazione per il registro provider di llmswitch.
    llmswitch_app_name: str = "personalities"
    #: Motori locali raggiungibili, da cui dipendono inferenza locale e CAG
    #: con KV-cache. Sono servizi remoti, non capacita' del cluster.
    local_engine_urls: List[str] = Field(default_factory=list)

    # --- accensione a richiesta del motore locale ---------------------------
    #: Un modello che occupa la memoria video la occupa anche mentre nessuno
    #: lo interroga. Questi comandi lo accendono quando serve e lo spengono
    #: quando non serve piu'. Vuoti: il motore si gestisce a mano, come in
    #: sviluppo con il server gia' avviato.
    #:
    #: **Sono comandi eseguiti dal worker con i suoi diritti**, e per questo
    #: arrivano solo di qui — dall'ambiente, in Kubernetes da un Secret. Non
    #: esiste endpoint ne' pagina che li cambi: la console puo' chiedere
    #: «accendi» o «spegni», non *cosa* eseguire. In produzione sono un `ssh`
    #: verso la macchina dei modelli, con una chiave a comando forzato per
    #: ciascuna delle due azioni: cosi' quella chiave non puo' fare altro.
    local_engine_start: str = ""
    local_engine_stop: str = ""
    #: Dove si chiede al motore se e' pronto. Senza, l'accensione si fida
    #: dell'uscita del comando — che dice se e' partito, non se risponde.
    local_engine_health_url: str = ""
    #: Quanto attendere che risponda dopo l'avvio. Un modello da ventisette
    #: miliardi di parametri si carica in decine di secondi, non in tre.
    local_engine_wait_s: float = 180.0
    #: Dopo quanta inattivita' spegnerlo. Zero: mai.
    local_engine_idle_s: float = 900.0

    #: Endpoint predefinito per la generazione. LM Studio, vLLM e OpenAI
    #: parlano lo stesso dialetto, quindi cambiare fornitore e' cambiare questo
    #: indirizzo — finche' la fase 1 non porta il registro di llmswitch.
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_model: str = ""          # vuoto: si usa il primo modello caricato
    llm_api_key: str = "non-serve-in-locale"
    #: Oltre questo tempo senza un singolo token la richiesta viene interrotta.
    #: E' un timeout fra i token, non sulla durata totale: una risposta lunga e'
    #: legittima, un silenzio di due minuti no.
    llm_stream_timeout: float = 120.0

    #: Quale fornitore genera le risposte: `openai-compatible` (OpenAI, LM
    #: Studio, vLLM, llama.cpp e affini) o `anthropic`.
    llm_provider: str = "openai-compatible"
    #: Che motore sta dietro `llm_base_url`: decide la `ContextStrategy`.
    #: `auto` lo deduce dall'indirizzo; si imposta a mano quando l'indirizzo
    #: non basta a riconoscerlo — un vLLM dietro un nome di dominio qualunque.
    #: Valori: auto, openai, vllm, llamacpp, lmstudio, other.
    llm_engine: str = "auto"
    #: Slot paralleli del server llama.cpp (`--parallel`). Serve a mandare lo
    #: stesso prefisso sempre allo stesso slot, dove la KV-cache e' calda.
    llm_kv_slots: int = 1

    #: Anthropic, quando `llm_provider = anthropic`.
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    #: Il Messages API pretende `max_tokens`: questo e' il valore quando la
    #: versione della personalita' non ne indica uno.
    anthropic_max_tokens: int = 2048

    #: Vettorizzazione. Separata dalla generazione perche' i due servizi
    #: possono stare su macchine diverse: l'embedding e' leggero e conviene
    #: tenerlo vicino al database, la generazione vuole l'acceleratore.
    embedding_base_url: str = "http://127.0.0.1:1234/v1"
    #: Multilingue e a 1024 dimensioni, come la colonna `vector` dello schema.
    #: Cambiarlo richiede una migrazione e il ricalcolo dell'indice: non e' una
    #: preferenza, e' una decisione di piattaforma.
    embedding_model: str = "text-embedding-qwen3-embedding-0.6b"

    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:3001"]
    )

    def validate_production(self) -> List[str]:
        """Impostazioni che in produzione sarebbero un difetto.

        Restituisce i problemi invece di sollevare: chi avvia il servizio
        decide se rifiutarsi di partire o soltanto segnalare.
        """
        problems: List[str] = []
        if self.environment == "prod":
            if self.auth_disabled:
                problems.append(
                    "PERSONA_AUTH_DISABLED e' attivo in produzione: ogni endpoint "
                    "sarebbe accessibile senza credenziali"
                )
            if "*" in self.cors_origins:
                problems.append("CORS aperto a qualunque origine in produzione")
            if "localhost" in self.database_url:
                problems.append("il database punta a localhost in produzione")
            if self.billing_provider == "mock":
                problems.append(
                    "fornitore di pagamento simulato in produzione: chiunque "
                    "potrebbe attivare qualunque piano senza pagare. Finché un "
                    "fornitore vero non c'è, PERSONA_BILLING_PROVIDER=disattivato"
                )
            # Senza fornitore non si accetta nessun evento, e il segreto non
            # firma niente: pretenderlo costringerebbe a inventarne uno.
            if (
                self.billing_provider != "disattivato"
                and self.billing_webhook_secret == "segreto-di-sviluppo-da-cambiare"
            ):
                problems.append(
                    "segreto dei webhook di pagamento lasciato al valore di "
                    "sviluppo: chiunque potrebbe firmare un pagamento finto"
                )
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
