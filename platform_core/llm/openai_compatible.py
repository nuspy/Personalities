"""Fornitore per gli endpoint in dialetto OpenAI.

Uno solo per LM Studio, vLLM, llama.cpp in modalità server, OpenAI stesso e i
servizi compatibili: parlano tutti lo stesso protocollo, e le differenze che
contano — quali campi rispettano, cosa riportano sul consumo — sono
differenze di comportamento, non di forma.

Le due asprezze gestite qui hanno entrambe una storia:

**Il silenzio del modello che ragiona.** I modelli con ragionamento esplicito
consumano il budget pensando e, se si esaurisce, chiudono con `finish_reason:
length` e contenuto vuoto. Non è un errore di rete: è una risposta valida e
inutile. Senza distinguerla, si conclude che la generazione è riuscita e si
salva una risposta vuota.

**Lo stream che si ferma senza chiudersi.** Una connessione che smette di
emettere token senza mai chiudersi terrebbe la richiesta appesa finché il
client non rinuncia. Il timeout è fra un token e l'altro, non sulla durata
complessiva: una risposta lunga è legittima, un silenzio prolungato no.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ..settings import Settings, get_settings
from .base import (
    GenerationError, GenerationRequest, LLMProvider, StreamChunk,
    TruncatedResponse, Usage,
)
from .json_mode import (
    DIALETTI, TENTATIVI_DI_BUDGET, TETTO_TOKEN, e_rifiuto_del_formato,
    estrai_json, formato_risposta, prossimo_budget,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = "",
        timeout: Optional[float] = None,
        settings: Optional[Settings] = None,
        name: str = "openai-compatibile",
    ) -> None:
        settings = settings or get_settings()
        self.name = name
        self._base_url = (base_url or settings.llm_base_url).rstrip("/")
        self._api_key = api_key or settings.llm_api_key
        self._default_model = default_model or settings.llm_model
        self._timeout = timeout or settings.llm_stream_timeout
        self._modello_risolto: Optional[str] = None
        #: Il dialetto JSON che questo server accetta. Si scopre alla prima
        #: richiesta e poi non si ridiscute: rinegoziarlo ogni volta
        #: costerebbe una richiesta fallita per ogni chiamata.
        self._dialetto_json: Optional[str] = None
        self._dialetto_negoziato = False

    # -- interrogazione ----------------------------------------------------

    async def available_models(self) -> List[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self._base_url}/models", headers=self._headers(),
            )
            response.raise_for_status()
            return [m["id"] for m in response.json().get("data", [])]

    async def resolve_model(self) -> str:
        """Il modello da usare, chiedendo al server se non è configurato.

        Con LM Studio è la situazione normale: l'utente carica un modello
        dall'interfaccia e il nome cambia a ogni cambio. Pretendere che sia
        scritto in configurazione significa una modifica al deploy ogni volta.
        """
        if self._default_model:
            return self._default_model
        if self._modello_risolto:
            return self._modello_risolto

        modelli = await self.available_models()
        if not modelli:
            raise GenerationError(
                f"nessun modello caricato su {self._base_url}: "
                f"caricane uno nel motore locale, o imposta PERSONA_LLM_MODEL"
            )
        self._modello_risolto = modelli[0]
        logger.info("Modello dedotto dal server: %s", self._modello_risolto)
        return self._modello_risolto

    # -- generazione -------------------------------------------------------

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        payload = await self._build_payload(request, stream=True)

        testo_accumulato: List[str] = []
        usage: Optional[Usage] = None
        finish_reason = ""

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as response:
                if response.status_code >= 400:
                    corpo = (await response.aread()).decode("utf-8", "replace")
                    raise GenerationError(
                        f"{response.status_code} da {self._base_url}: {corpo[:500]}"
                    )

                async for riga in response.aiter_lines():
                    if not riga or not riga.startswith("data:"):
                        continue
                    dato = riga[5:].strip()
                    if dato == "[DONE]":
                        break

                    try:
                        evento = json.loads(dato)
                    except json.JSONDecodeError:
                        # Un frammento malformato non giustifica far cadere
                        # l'intera risposta: si annota e si prosegue.
                        logger.debug("Frammento non interpretabile: %s", dato[:120])
                        continue

                    if (uso := evento.get("usage")):
                        usage = self._parse_usage(uso, payload["model"])

                    for scelta in evento.get("choices", []):
                        delta = scelta.get("delta") or {}
                        if scelta.get("finish_reason"):
                            finish_reason = scelta["finish_reason"]

                        # Nome del campo diverso da un motore all'altro: LM
                        # Studio usa `reasoning`, altri `reasoning_content`.
                        pensiero = delta.get("reasoning") or delta.get(
                            "reasoning_content"
                        ) or ""
                        if pensiero:
                            yield StreamChunk(reasoning=pensiero)

                        if (testo := delta.get("content")):
                            testo_accumulato.append(testo)
                            yield StreamChunk(text=testo)

        completo = "".join(testo_accumulato)
        if finish_reason == "length" and not completo.strip():
            raise TruncatedResponse(
                "il modello ha esaurito il budget senza produrre testo: "
                "probabilmente ha speso tutti i token a ragionare. "
                "Aumenta max_tokens o usa un modello senza ragionamento esteso.",
                partial=completo,
                usage=usage,
            )

        yield StreamChunk(done=True, usage=usage, finish_reason=finish_reason)

    async def complete(self, request: GenerationRequest) -> str:
        pezzi: List[str] = []
        async for chunk in self.stream(request):
            if chunk.text:
                pezzi.append(chunk.text)
        return "".join(pezzi)

    async def complete_json(self, request: GenerationRequest) -> Any:
        """Genera e interpreta una risposta JSON.

        Mette insieme le tre difese di `json_mode`: negozia il dialetto che il
        server accetta, alza il budget se il modello lo esaurisce ragionando, e
        interpreta con tolleranza ciò che torna.

        Non è streaming perché non avrebbe senso: un JSON parziale non si può
        mostrare, e chi chiama aspetta comunque la struttura completa.
        """
        dialetti = [self._dialetto_json] if self._dialetto_negoziato else list(DIALETTI)
        ultimo_errore: Optional[Exception] = None

        for dialetto in dialetti:
            try:
                testo = await self._genera_con_budget(request, dialetto)
            except GenerationError as exc:
                if not e_rifiuto_del_formato(exc):
                    raise
                ultimo_errore = exc
                logger.info(
                    "Il server non accetta response_format '%s': provo il successivo",
                    dialetto,
                )
                continue

            if not self._dialetto_negoziato:
                self._dialetto_json = dialetto
                self._dialetto_negoziato = True
                logger.info(
                    "Modalità JSON negoziata: %s",
                    f"response_format '{dialetto}'" if dialetto
                    else "nessun vincolo formale",
                )
            return estrai_json(testo)

        raise GenerationError(
            f"nessuna modalità JSON accettata dal server ({ultimo_errore})"
        )

    async def _genera_con_budget(
        self, request: GenerationRequest, dialetto: Optional[str]
    ) -> str:
        """Riprova con più spazio quando il modello esaurisce il budget."""
        budget = request.max_tokens or 4096

        for tentativo in range(TENTATIVI_DI_BUDGET + 1):
            richiesta = replace(
                request,
                max_tokens=budget,
                extra={**request.extra, **_formato(dialetto)},
            )
            try:
                return await self.complete(richiesta)
            except TruncatedResponse as exc:
                if tentativo == TENTATIVI_DI_BUDGET or budget >= TETTO_TOKEN:
                    raise GenerationError(
                        f"il modello non produce una risposta nemmeno con "
                        f"{budget} token: {exc}"
                    ) from exc
                budget = prossimo_budget(budget)
                logger.info(
                    "Risposta troncata: riprovo con max_tokens=%d", budget,
                )

        raise GenerationError("budget esaurito")  # pragma: no cover

    # -- interni -----------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _build_payload(
        self, request: GenerationRequest, *, stream: bool
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model or await self.resolve_model(),
            "messages": [m.to_dict() for m in request.messages],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if stream:
            # Senza, molti server non riportano affatto il consumo negli
            # eventi di streaming, e la contabilità dei token resta vuota
            # proprio nel percorso che si usa sempre.
            payload["stream_options"] = {"include_usage": True}
        payload.update(request.extra)
        return payload

    @staticmethod
    def _parse_usage(uso: Dict[str, Any], model: str) -> Usage:
        dettagli = uso.get("prompt_tokens_details") or {}
        return Usage(
            prompt_tokens=uso.get("prompt_tokens", 0),
            completion_tokens=uso.get("completion_tokens", 0),
            cached_tokens=dettagli.get("cached_tokens", 0),
            total_tokens=uso.get("total_tokens", 0),
            model=model,
        )


def _formato(dialetto: Optional[str]) -> Dict[str, Any]:
    """Il frammento di payload che impone il formato, o niente."""
    formato = formato_risposta(dialetto)
    return {"response_format": formato} if formato else {}
