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
    BUDGET_RAGIONAMENTO, DIALETTI, TENTATIVI_DI_BUDGET, TETTO_TOKEN,
    e_rifiuto_del_formato,
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
        motore: Optional[str] = None,
        ragiona: bool = False,
    ) -> None:
        settings = settings or get_settings()
        self.name = name
        self._base_url = (base_url or settings.llm_base_url).rstrip("/")
        #: Il motore dietro l'indirizzo: decide quale `ContextStrategy` si usa
        #: e quali campi di cache si possono mandare. Dichiarato qui perché è
        #: il fornitore a sapere a cosa sta parlando.
        self.motore = motore or riconosci_motore(self._base_url, settings.llm_engine)
        #: Se il modello ragiona prima di rispondere. Alza il pavimento del
        #: budget nelle richieste strutturate: vedi `BUDGET_RAGIONAMENTO`.
        self.ragiona = ragiona
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
        # Il pavimento per i modelli che ragionano lo applica il costruttore
        # del payload, che vale anche per le chiamate non strutturate: qui
        # basta la scala dei tentativi.
        budget = request.max_tokens or 4096

        for tentativo in range(TENTATIVI_DI_BUDGET + 1):
            richiesta = replace(
                request,
                max_tokens=budget,
                extra={**request.extra, **_formato(dialetto)},
            )
            try:
                testo = await self.complete(richiesta)
            except TruncatedResponse as exc:
                motivo = str(exc)
            else:
                # Una risposta **con** contenuto ma tagliata a metà non
                # solleva `TruncatedResponse`: quella segnala il caso in cui il
                # modello ha speso tutto a ragionare e non ha scritto nulla.
                # Qui il testo c'è, e per il testo libero sarebbe utilizzabile
                # — ma un JSON interrotto a metà graffa non lo è.
                #
                # È il difetto che ha fatto fallire la digestione del primo
                # corpus: quattro lotti su dodici persi, con in log un JSON
                # che cominciava bene e si fermava a metà. Il budget adattivo
                # c'era e non scattava mai.
                if _sembra_troncato(testo):
                    motivo = "JSON incompleto: struttura non chiusa"
                else:
                    return testo

            if tentativo == TENTATIVI_DI_BUDGET or budget >= TETTO_TOKEN:
                raise GenerationError(
                    f"il modello non produce una risposta utilizzabile nemmeno "
                    f"con {budget} token: {motivo}"
                )
            budget = prossimo_budget(budget)
            logger.info(
                "Risposta inutilizzabile (%s): riprovo con max_tokens=%d",
                motivo, budget,
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
            # Il pavimento per i modelli che ragionano vale su **ogni**
            # chiamata, non solo su quelle strutturate: il ragionamento
            # consuma lo stesso budget della risposta, e un tetto calcolato
            # sulla lunghezza attesa della risposta lo esaurisce prima che
            # il modello cominci a scrivere. Il risultato è testo vuoto —
            # nessun errore — e chi chiama ricade sul suo ripiego a ogni
            # tentativo, senza che nulla sembri rotto.
            #
            # È un tetto, non un obiettivo: chi ha finito si ferma, e su un
            # modello che non ragiona la riga non fa nulla.
            payload["max_tokens"] = (
                max(request.max_tokens, BUDGET_RAGIONAMENTO)
                if self.ragiona else request.max_tokens
            )
        if request.stop:
            payload["stop"] = request.stop
        if stream:
            # Senza, molti server non riportano affatto il consumo negli
            # eventi di streaming, e la contabilità dei token resta vuota
            # proprio nel percorso che si usa sempre.
            payload["stream_options"] = {"include_usage": True}
        payload.update(self._campi_di_cache(request))
        payload.update(request.extra)
        return payload

    def _campi_di_cache(self, request: GenerationRequest) -> Dict[str, Any]:
        """Traduce l'indicazione di cache nei campi che questo motore capisce.

        Solo quelli che il motore conosce, e solo per lui: vLLM rifiuta o
        segnala i campi sconosciuti, e mandare `cache_prompt` a OpenAI o
        `prompt_cache_key` a un llama.cpp sarebbe nel migliore dei casi
        inutile e nel peggiore un 400 su ogni risposta.
        """
        suggerimento = request.cache
        if suggerimento is None or suggerimento.modo == "none":
            return {}

        if self.motore == "llamacpp" and suggerimento.modo == "kv":
            campi: Dict[str, Any] = {"cache_prompt": True}
            if suggerimento.slot is not None:
                campi["id_slot"] = suggerimento.slot
            return campi

        if self.motore == "openai" and suggerimento.modo == "prompt":
            # Il caching di OpenAI è automatico sopra i 1024 token; la chiave
            # instrada le richieste con lo stesso prefisso verso la stessa
            # macchina, dove il prefisso è già in cache.
            return {"prompt_cache_key": suggerimento.chiave}

        # vLLM e LM Studio riusano il prefisso da sé: non serve nessun campo.
        return {}

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


def riconosci_motore(base_url: str, configurato: str = "auto") -> str:
    """Che motore sta dietro un indirizzo OpenAI-compatibile.

    Dall'indirizzo si riconoscono con sicurezza solo i casi comuni: OpenAI dal
    dominio, LM Studio e llama.cpp dalle loro porte predefinite. Un vLLM o un
    llama.cpp dietro un nome qualunque restano `other` — cioè prezzo pieno —
    finché qualcuno non lo dichiara in `PERSONA_LLM_ENGINE`. Indovinare un
    motore con cache e sbagliare manderebbe campi che il server rifiuta.
    """
    if configurato and configurato != "auto":
        return configurato

    indirizzo = base_url.lower()
    if "api.openai.com" in indirizzo:
        return "openai"
    if ":1234" in indirizzo:
        return "lmstudio"
    if ":8080" in indirizzo and ("127.0.0.1" in indirizzo or "localhost" in indirizzo):
        return "llamacpp"
    return "other"


def _formato(dialetto: Optional[str]) -> Dict[str, Any]:
    """Il frammento di payload che impone il formato, o niente."""
    formato = formato_risposta(dialetto)
    return {"response_format": formato} if formato else {}


def _sembra_troncato(testo: str) -> bool:
    """Vero se il testo è un JSON cominciato e mai chiuso.

    Conta le parentesi fuori dalle stringhe: se ne restano di aperte, la
    risposta si è interrotta. Non è un parser — non serve — ma distingue il
    caso che conta da un modello che ha semplicemente risposto in prosa,
    dove riprovare con più spazio non cambierebbe nulla.
    """
    pulito = testo.strip()
    if not pulito.startswith(("{", "[", "```")):
        return False

    aperte = 0
    in_stringa = False
    fuga = False

    for carattere in pulito:
        if fuga:
            fuga = False
            continue
        if carattere == "\\":
            fuga = True
        elif carattere == '"':
            in_stringa = not in_stringa
        elif not in_stringa and carattere in "{[":
            aperte += 1
        elif not in_stringa and carattere in "}]":
            aperte -= 1

    return aperte > 0 or in_stringa
