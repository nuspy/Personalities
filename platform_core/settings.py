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
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
