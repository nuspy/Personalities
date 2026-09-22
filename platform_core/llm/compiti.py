"""Quale modello serve quale compito.

Un solo modello per tutto è comodo finché i compiti non divergono, e
divergono presto. Classificare cinquemila passaggi di un corpus vuole un
modello economico e instancabile; rispondere a un utente vuole il migliore
che ci si possa permettere; verificare se una risposta sia ancorata ai
passaggi vuole un modello **diverso da quello che l'ha scritta**, o il
giudice finisce per assolvere sé stesso.

Da qui i quattro compiti. Non sono livelli di qualità: sono lavori con
requisiti diversi, e la scelta di ciascuno è indipendente dalle altre.

**Come si configura, e perché così.** Gli indirizzi e le chiavi stanno
nell'ambiente — in Kubernetes in un Secret — e formano un elenco chiuso di
modelli, ciascuno con un nome. La console non li crea e non li modifica:
sceglie, fra quei nomi, chi serve quale compito. La ragione è che un
endpoint modificabile dall'interfaccia è traffico dirottabile verso una
macchina qualunque, con le chiavi appresso; scegliere fra un elenco già
approvato non lo è.

Chi non ha un'assegnazione ricade sul modello predefinito. Ricade e non
fallisce: un compito senza modello assegnato è la condizione normale di un
impianto appena installato, e una piattaforma che non risponde finché
qualcuno non ha compilato quattro caselle è una piattaforma che sembra
rotta.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Compito(str, Enum):
    """I lavori per cui si interroga un modello."""

    #: Le risposte agli utenti.
    CONVERSAZIONE = "conversazione"
    #: Il lavoro di fondo sui testi: classificare i passaggi di un corpus,
    #: ripulire l'apparato del curatore, riconoscere la provenienza, ed
    #: estrarre le memorie da una conversazione chiusa. Sta insieme perché è
    #: la stessa cosa — leggere del testo e ricavarne dati — e perché ha gli
    #: stessi requisiti: nessuno aspetta, e le chiamate sono tante.
    DIGESTIONE = "digestione"
    #: RAG e CAG: costruire la domanda con cui interrogare i corpora, e
    #: rielaborare i passaggi recuperati prima che entrino nel contesto.
    RECUPERO = "recupero"
    #: Groundcheck e guardrail: il giudice che verifica le affermazioni.
    GIUDIZIO = "giudizio"

    @property
    def label(self) -> str:
        return _ETICHETTE[self]

    @property
    def descrizione(self) -> str:
        return _DESCRIZIONI[self]


_ETICHETTE: Dict[Compito, str] = {
    Compito.CONVERSAZIONE: "Risposte agli utenti",
    Compito.DIGESTIONE: "Digestione e ingestione dei corpora",
    Compito.RECUPERO: "Recupero: domande e rielaborazione",
    Compito.GIUDIZIO: "Giudizio: fondatezza e guardrail",
}

_DESCRIZIONI: Dict[Compito, str] = {
    Compito.CONVERSAZIONE: (
        "Chi parla con gli utenti. È il modello che si sente: qui la qualità "
        "si nota a ogni frase."
    ),
    Compito.DIGESTIONE: (
        "Classifica i passaggi di un corpus, ne toglie l'apparato editoriale, "
        "ed estrae le memorie a fine conversazione. Lavoro di fondo, migliaia "
        "di chiamate: conta più il costo della finezza."
    ),
    Compito.RECUPERO: (
        "Costruisce la domanda con cui si interrogano i corpora e rielabora i "
        "passaggi trovati. Una chiamata in più per risposta: si accende per "
        "personalità, non per tutte."
    ),
    Compito.GIUDIZIO: (
        "Verifica che le affermazioni di fatto stiano nei passaggi. "
        "**Conviene che sia un modello diverso** da quello che ha risposto: "
        "un giudice che è anche l'autore assolve sé stesso."
    ),
}


class ModelloConfigurato(BaseModel):
    """Un modello raggiungibile, con il nome con cui la console lo chiama."""

    #: Il nome che si vede in console e che l'assegnazione registra. Stabile:
    #: cambiarlo fa ricadere sui predefiniti i compiti che lo citavano.
    nome: str = Field(min_length=1, max_length=60)
    #: `openai-compatible` (OpenAI, LM Studio, vLLM, llama.cpp) o `anthropic`.
    provider: str = "openai-compatible"
    base_url: str = ""
    #: Vuoto: il primo modello che il server ha caricato.
    model: str = ""
    api_key: str = ""
    #: Che motore c'è dietro l'indirizzo: decide la `ContextStrategy`, cioè
    #: se il prefisso stabile viene riusato dalla KV-cache, scontato dal
    #: prompt caching del fornitore, o niente. Vuoto lo deduce dall'indirizzo;
    #: si indica quando l'indirizzo non basta — un llama.cpp su una porta
    #: qualunque somiglia a qualunque altra cosa.
    #: Valori: vuoto, `openai`, `vllm`, `llamacpp`, `lmstudio`, `other`.
    motore: str = ""
    #: Come si presenta in console. Non ha effetto sul comportamento.
    descrizione: str = ""
    #: Dichiara che il modello non rifiuta nulla — un modello «abliterato» ha
    #: perso la capacità di dire di no. Non impedisce nessuna assegnazione:
    #: è l'amministratore a decidere. Serve perché la scelta sia informata
    #: invece che accidentale, e perché resti scritto cosa si è scelto.
    senza_filtri: bool = False


#: Il nome che prende il modello ricavato dalle vecchie impostazioni, quando
#: nessun elenco è configurato.
PREDEFINITO = "predefinito"


class RegistroModelli:
    """L'elenco dei modelli e chi serve cosa.

    Costruisce i fornitori una volta sola per modello: l'oggetto ricorda
    quale modello il server ha caricato, e ricrearlo a ogni chiamata
    significherebbe chiedere di nuovo l'elenco dei modelli prima di ogni
    singola risposta.
    """

    def __init__(
        self,
        modelli: List[ModelloConfigurato],
        *,
        assegnazioni: Optional[Dict[Compito, str]] = None,
    ) -> None:
        self._modelli = {m.nome: m for m in modelli}
        self._assegnazioni: Dict[Compito, str] = dict(assegnazioni or {})
        self._fornitori: Dict[str, object] = {}
        #: Quando le assegnazioni sono state lette dal database. Zero: mai.
        self._lette_a = 0.0

    # ------------------------------------------------------------- elenco

    @property
    def modelli(self) -> List[ModelloConfigurato]:
        return list(self._modelli.values())

    def esiste(self, nome: str) -> bool:
        return nome in self._modelli

    def configurazione(self, nome: str) -> Optional[ModelloConfigurato]:
        return self._modelli.get(nome)

    # -------------------------------------------------------- assegnazioni

    def assegna(self, compito: Compito, nome: Optional[str]) -> None:
        """Registra o toglie un'assegnazione. `None` fa tornare al predefinito."""
        if nome is None:
            self._assegnazioni.pop(compito, None)
            return
        if nome not in self._modelli:
            raise ValueError(f"modello sconosciuto: {nome!r}")
        self._assegnazioni[compito] = nome

    def aggiorna(self, assegnazioni: Dict[Compito, str]) -> None:
        """Sostituisce tutte le assegnazioni con quelle lette dal database.

        Sostituisce e non fonde: un compito tolto dalla tabella deve tornare
        al predefinito, e una fusione lo lascerebbe assegnato per sempre al
        modello di prima.
        """
        import time

        self._assegnazioni = {
            compito: nome for compito, nome in assegnazioni.items() if nome
        }
        self._lette_a = time.monotonic()

    def da_rileggere(self, ttl: float = 30.0) -> bool:
        """Se vale la pena rileggere le assegnazioni dal database.

        Con più repliche dell'API, una modifica fatta su una deve arrivare
        alle altre: nessuna se ne accorgerebbe da sola. Mezzo minuto di
        ritardo è accettabile — è un cambio di configurazione, non una
        risposta — e costa una riga letta ogni trenta secondi per replica.
        """
        import time

        return time.monotonic() - self._lette_a > ttl

    def nome_per(self, compito: Compito) -> str:
        """Il nome del modello che serve questo compito, assegnato o no."""
        assegnato = self._assegnazioni.get(compito)
        if assegnato and assegnato in self._modelli:
            return assegnato
        if assegnato:
            # Un modello tolto dall'ambiente mentre l'assegnazione restava:
            # va detto, perché il compito sta girando su un altro modello e
            # nessuno se ne accorgerebbe dai risultati.
            logger.warning(
                "Il compito «%s» era assegnato a «%s», che non è più fra i "
                "modelli configurati: si usa il predefinito",
                compito.value, assegnato,
            )
        if PREDEFINITO in self._modelli:
            return PREDEFINITO
        return next(iter(self._modelli), "")

    def per(self, compito: Compito):
        """Il fornitore che serve questo compito."""
        nome = self.nome_per(compito)
        if not nome:
            raise RuntimeError(
                "nessun modello configurato: PERSONA_MODELLI è vuoto e non "
                "esiste nemmeno un modello predefinito"
            )
        return self._fornitore(nome)

    def stato(self) -> List[dict]:
        """Cosa mostrare in console: i compiti, chi li serve, e perché."""
        return [
            {
                "compito": compito.value,
                "label": compito.label,
                "descrizione": compito.descrizione,
                "assegnato": self._assegnazioni.get(compito),
                "in_uso": self.nome_per(compito),
                "senza_filtri": bool(
                    (m := self._modelli.get(self.nome_per(compito)))
                    and m.senza_filtri
                ),
            }
            for compito in Compito
        ]

    # ------------------------------------------------------------- interno

    def _fornitore(self, nome: str):
        if nome in self._fornitori:
            return self._fornitori[nome]

        configurazione = self._modelli[nome]
        if configurazione.provider == "anthropic":
            from .anthropic import AnthropicProvider

            fornitore = AnthropicProvider(
                base_url=configurazione.base_url or None,
                api_key=configurazione.api_key or None,
                default_model=configurazione.model,
            )
        else:
            from .openai_compatible import OpenAICompatibleProvider

            fornitore = OpenAICompatibleProvider(
                base_url=configurazione.base_url or None,
                api_key=configurazione.api_key or None,
                default_model=configurazione.model,
                motore=configurazione.motore or None,
                # Il nome finisce nelle tracce: senza, due modelli diversi
                # comparirebbero entrambi come «openai-compatibile» e non si
                # saprebbe quale ha prodotto quale risposta.
                name=nome,
            )

        self._fornitori[nome] = fornitore
        return fornitore


def modelli_da_impostazioni() -> List[ModelloConfigurato]:
    """L'elenco dei modelli, dall'ambiente.

    Senza elenco se ne ricava uno di un elemento dalle vecchie impostazioni
    `PERSONA_LLM_*`: un impianto che funzionava prima continua a funzionare
    senza toccare nulla, e i quattro compiti girano tutti su quel modello
    finché qualcuno non decide altrimenti.
    """
    from ..settings import get_settings

    s = get_settings()
    if s.modelli:
        return list(s.modelli)

    return [
        ModelloConfigurato(
            nome=PREDEFINITO,
            provider=s.llm_provider,
            base_url=(
                s.anthropic_base_url if s.llm_provider == "anthropic" else s.llm_base_url
            ),
            model=(
                s.anthropic_model if s.llm_provider == "anthropic" else s.llm_model
            ),
            api_key=(
                s.anthropic_api_key if s.llm_provider == "anthropic" else s.llm_api_key
            ),
            descrizione="Dalle impostazioni PERSONA_LLM_*",
        )
    ]
