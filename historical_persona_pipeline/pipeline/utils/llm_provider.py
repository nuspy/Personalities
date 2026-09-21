"""Provider LLM con ritentativi e gestione corretta della modalita' JSON.

Tre difetti della versione precedente:

1. **nessun ritentativo**: un errore di rete o un 503 momentaneo bruciava
   l'esempio in corso, e su una generazione da mille conversazioni con un
   modello locale succede di continuo;
2. **`response_format: json_object` con prompt che chiedevano un array**:
   la modalita' JSON di OpenAI impone un *oggetto* alla radice, quindi il
   modello restituiva un oggetto e il parser, che si aspettava una lista,
   tornava vuoto senza segnalare nulla;
3. **timeout di 60 secondi**: sufficiente per un servizio remoto, non per un
   modello locale che genera cinque conversazioni.
"""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from .tls import enable_system_certificates

logger = logging.getLogger(__name__)

enable_system_certificates()

DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.8

# I modelli di ragionamento spendono nel pensiero la maggior parte del budget:
# 4096 token bastano a una risposta diretta, non a un ragionamento piu' una
# risposta. Il valore iniziale e' generoso e cresce se non basta.
DEFAULT_MAX_TOKENS = 8192
MAX_TOKENS_CEILING = 32768
MAX_BUDGET_RETRIES = 2

# Codici su cui ha senso riprovare: sovraccarico o errore temporaneo.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Errore non recuperabile dopo i ritentativi previsti."""


class TruncatedResponse(LLMError):
    """La risposta e' arrivata vuota perche' il budget di token e' finito.

    Distinta dagli altri errori perche' ha una cura precisa — alzare
    `max_tokens` — mentre gli altri richiedono di cambiare qualcos'altro.
    """


def _explain_empty_content(choice: Dict[str, Any], message: Dict[str, Any],
                           usage: Dict[str, Any]) -> str:
    """Spiega perche' il modello ha restituito contenuto vuoto."""
    finish = choice.get("finish_reason", "?")
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    completion = usage.get("completion_tokens", 0)

    if finish == "length" and reasoning:
        return (
            f"il modello ha speso {reasoning} dei {completion} token disponibili "
            "nel ragionamento interno, senza arrivare alla risposta"
        )
    if finish == "length":
        return f"risposta troncata dopo {completion} token (max_tokens insufficiente)"
    if message.get("reasoning_content"):
        return f"contenuto vuoto ma presente un ragionamento interno (finish_reason={finish})"
    return f"il modello ha restituito contenuto vuoto (finish_reason={finish})"


def _response_detail(response) -> str:
    """Messaggio d'errore leggibile dal corpo della risposta."""
    if response is None:
        return ""
    try:
        data = response.json()
    except Exception:
        return (response.text or "")[:300].strip()

    if isinstance(data, dict):
        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        return str(error)[:300]
    return str(data)[:300]


def _is_response_format_rejection(exc: Exception) -> bool:
    """Vero se il server ha rifiutato proprio il vincolo di formato JSON."""
    text = str(exc).lower()
    return "response_format" in text or "json_schema" in text


def _json_response_format(dialect: Optional[str]) -> Optional[Dict[str, Any]]:
    """Payload `response_format` per il dialetto indicato."""
    if dialect == "json_object":
        return {"type": "json_object"}
    if dialect == "json_schema":
        # Schema deliberatamente permissivo: serve a ottenere JSON valido,
        # non a imporre una forma — le risposte attese hanno chiavi diverse
        # a seconda del punto della pipeline che le richiede.
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "risposta",
                "strict": False,
                "schema": {"type": "object", "additionalProperties": True},
            },
        }
    return None


class LLMProvider(ABC):
    def __init__(
        self,
        model: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        ...

    def health_check(self) -> tuple[bool, str]:
        """Verifica rapida che il servizio risponda.

        Chiamata prima di una generazione lunga: accorgersi che l'endpoint e'
        spento dopo venti minuti di tentativi falliti e' il modo peggiore.
        """
        try:
            self.generate("Rispondi con: ok", system_prompt=None, json_mode=False)
            return True, "raggiungibile"
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------ ritentativi

    def _with_retries(self, operation, description: str) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                last_error = exc
                if status not in RETRYABLE_STATUS:
                    # Il corpo della risposta contiene il motivo del rifiuto:
                    # senza, un 400 non dice nulla e costringe a indovinare.
                    detail = _response_detail(exc.response)
                    raise LLMError(
                        f"{description}: errore {status} non recuperabile"
                        + (f" — {detail}" if detail else "")
                    ) from exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
            except TruncatedResponse:
                # Deve arrivare intatta a `_generate_with_budget`, che sa
                # come rimediare: riavvolgerla in un LLMError generico
                # perderebbe il tipo e con esso il ritentativo col budget
                # maggiorato. Ritentare qui sarebbe inutile — lo stesso
                # tetto di token produrrebbe lo stesso troncamento.
                raise
            except Exception as exc:
                raise LLMError(f"{description}: {exc}") from exc

            if attempt < self.max_retries:
                # Backoff esponenziale con jitter: piu' tentativi in parallelo
                # non devono riprovare tutti nello stesso istante.
                delay = min(30.0, 2.0 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"{description}: tentativo {attempt}/{self.max_retries} fallito "
                    f"({last_error}). Riprovo fra {delay:.1f}s"
                )
                time.sleep(delay)

        raise LLMError(f"{description}: falliti {self.max_retries} tentativi ({last_error})")


class OpenAICompatibleProvider(LLMProvider):
    """Qualsiasi endpoint con API in stile OpenAI: LM Studio, Ollama, Nebius, Groq."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(model, timeout, max_retries, temperature)
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        # Dialetto di modalita' JSON accettato dal server: si scopre alla
        # prima richiesta e poi si riusa, invece di sprecare un tentativo
        # fallito per ogni chiamata.
        self._json_dialect: Optional[str] = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        base_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if not json_mode:
            return self._generate_with_budget(base_payload, f"LLM {self.model}")

        # I server non concordano su come si chiede una risposta JSON:
        # OpenAI e la maggior parte dei cloni usano `json_object`, LM Studio
        # recente accetta solo `json_schema` o `text`, altri non supportano
        # nulla. Si prova in ordine e si ricorda cio' che funziona; senza
        # vincolo formale il prompt chiede comunque JSON e il parser a valle
        # tollera il testo libero.
        dialects = [self._json_dialect] if self._json_dialect else ["json_object", "json_schema", None]

        last_error: Optional[Exception] = None
        for dialect in dialects:
            payload = dict(base_payload)
            response_format = _json_response_format(dialect)
            if response_format:
                payload["response_format"] = response_format

            try:
                result = self._generate_with_budget(payload, f"LLM {self.model}")
            except LLMError as exc:
                if not _is_response_format_rejection(exc):
                    raise
                last_error = exc
                logger.info(
                    f"Il server non accetta response_format '{dialect}': provo il formato successivo."
                )
                continue

            if self._json_dialect != dialect:
                self._json_dialect = dialect
                logger.info(
                    "Modalita' JSON negoziata: "
                    + (f"response_format '{dialect}'" if dialect else "nessun vincolo formale")
                )
            return result

        raise LLMError(
            f"Nessuna modalita' JSON accettata dal server ({last_error})"
        )

    def _post(self, payload: Dict[str, Any]) -> str:
        def call() -> str:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            try:
                choice = data["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError) as exc:
                raise LLMError(f"Risposta in formato inatteso: {data}") from exc

            content = message.get("content") or ""
            if content.strip():
                return content

            # Contenuto vuoto: il motivo va detto, altrimenti a valle si
            # vede solo una risposta scartata e non si capisce perche'.
            raise TruncatedResponse(
                _explain_empty_content(choice, message, data.get("usage", {}))
            )

        return self._with_retries(call, f"LLM {self.model}")

    def _generate_with_budget(
        self, payload: Dict[str, Any], description: str
    ) -> str:
        """Esegue la richiesta alzando il budget di token se viene esaurito.

        I modelli di ragionamento spendono la maggior parte dei token a
        pensare, e quello che pensano non finisce nella risposta: con un tetto
        troppo basso il ragionamento consuma tutto e `content` resta vuoto.
        Non e' un errore del modello ne' del prompt — e' un budget
        insufficiente, e la cura e' darne di piu'.
        """
        attempts = 0
        while True:
            try:
                return self._post(payload)
            except TruncatedResponse as exc:
                attempts += 1
                current = int(payload.get("max_tokens", self.max_tokens))
                if attempts > MAX_BUDGET_RETRIES or current >= MAX_TOKENS_CEILING:
                    raise LLMError(f"{description}: {exc}") from exc

                payload = dict(payload)
                payload["max_tokens"] = min(current * 3, MAX_TOKENS_CEILING)
                logger.info(
                    f"{description}: risposta troncata ({exc}). "
                    f"Riprovo con max_tokens={payload['max_tokens']}."
                )
                # Il budget piu' alto vale anche per le richieste successive:
                # se serve una volta, servira' ancora.
                self.max_tokens = payload["max_tokens"]


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(model, timeout, max_retries, temperature)
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "Pacchetto 'anthropic' non installato: pip install anthropic"
            ) from exc

        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if json_mode:
            # L'API non ha una modalita' JSON: si precompila l'inizio della
            # risposta con "{" perche' il modello prosegua in JSON.
            kwargs["messages"].append({"role": "assistant", "content": "{"})

        def call() -> str:
            response = self.client.messages.create(**kwargs)
            text = response.content[0].text
            return "{" + text if json_mode else text

        # Il client Anthropic ritenta gia' per conto suo; qui si copre il resto.
        return self._with_retries(call, f"Anthropic {self.model}")


class LLMFactory:
    # Alias storici mantenuti per compatibilita' con le configurazioni esistenti.
    OPENAI_COMPATIBLE_ALIASES = frozenset({
        "lm_studio", "openai_compatible", "openai", "nebius",
        "groq", "deepseek", "ollama", "glm4", "vllm", "together",
    })

    @staticmethod
    def create(config: Dict[str, Any]) -> LLMProvider:
        """Costruisce il provider dalla sezione `dataset` della configurazione."""
        provider_type = str(config.get("llm_provider", "lm_studio")).lower()

        common = {
            "timeout": int(config.get("llm_timeout", DEFAULT_TIMEOUT)),
            "max_retries": int(config.get("max_retries", DEFAULT_MAX_RETRIES)),
            "temperature": float(config.get("llm_temperature", DEFAULT_TEMPERATURE)),
            "max_tokens": int(config.get("llm_max_tokens", DEFAULT_MAX_TOKENS)),
        }

        if provider_type == "anthropic":
            api_key = config.get("llm_api_key")
            if not api_key:
                raise LLMError("Provider 'anthropic' senza chiave API (dataset.llm_api_key)")
            return AnthropicProvider(
                api_key=api_key,
                model=config.get("llm_model_name") or "claude-sonnet-5",
                **common,
            )

        if provider_type in LLMFactory.OPENAI_COMPATIBLE_ALIASES:
            base_url = config.get("llm_base_url") or "http://localhost:1234/v1"
            return OpenAICompatibleProvider(
                base_url=base_url,
                api_key=config.get("llm_api_key") or "not-needed",
                model=config.get("llm_model_name") or "local-model",
                **common,
            )

        raise LLMError(
            f"Provider LLM sconosciuto: '{provider_type}'. "
            f"Validi: anthropic, {', '.join(sorted(LLMFactory.OPENAI_COMPATIBLE_ALIASES))}"
        )


# Nome storico: alcuni moduli importano ancora LMStudioProvider.
LMStudioProvider = OpenAICompatibleProvider
