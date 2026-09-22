"""Le astrazioni della voce.

`TTSProvider` e `STTProvider` stanno dietro un `Protocol` come tutto il resto:
nessun modulo di dominio importa un SDK. La ragione qui è concreta — i
fornitori di sintesi cambiano spesso, hanno prezzi molto diversi e nessuno di
loro restituisce le stesse cose degli altri.

**Cosa deve restituire una sintesi, e perché non è solo audio.** Il client
deve poter animare una bocca e evidenziare la parola in corso, e per farlo gli
servono i tempi. Quasi nessun fornitore li dà: OpenAI no, Piper no, ElevenLabs
solo su alcuni piani. Per questo `Sintesi` li dichiara **opzionali** e chi la
costruisce dice da dove vengono — dal fornitore o da un allineamento fatto
dopo. Fingere che ci siano sempre significherebbe che il client anima su tempi
inventati, e un labiale fuori sincrono è peggio di una bocca ferma.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .visemi import Parola, Viseme


class SintesiNonDisponibile(RuntimeError):
    """Nessun fornitore di voce configurato, o non raggiungibile."""


@dataclass
class Sintesi:
    """Audio parlato, e quel che serve per animarlo."""

    #: I byte dell'audio. Il formato lo dice `media_type`.
    audio: bytes
    media_type: str = "audio/mpeg"
    durata: float = 0.0
    voce: str = ""
    modello: str = ""

    #: I tempi delle parole, quando si sanno. Vuoti non è un errore: significa
    #: che nessuno li ha misurati, e il client deve saperlo per non animare
    #: su valori inventati.
    parole: List[Parola] = field(default_factory=list)
    visemi: List[Viseme] = field(default_factory=list)

    #: Come sono stati ottenuti i tempi: `fornitore`, `allineamento`, oppure
    #: vuoto quando non ce ne sono. È la differenza fra un labiale affidabile
    #: e uno approssimato, e chi guarda l'animazione deve poterla conoscere.
    origine_tempi: str = ""

    @property
    def sincronizzabile(self) -> bool:
        return bool(self.visemi)

    def to_dict(self, *, con_audio: bool = False) -> Dict[str, Any]:
        import base64

        corpo: Dict[str, Any] = {
            "media_type": self.media_type,
            "durata": round(self.durata, 3),
            "voce": self.voce,
            "modello": self.modello,
            "origine_tempi": self.origine_tempi,
            "parole": [p.to_dict() for p in self.parole],
            "visemi": [v.to_dict() for v in self.visemi],
        }
        if con_audio:
            corpo["audio"] = base64.b64encode(self.audio).decode("ascii")
        return corpo


@dataclass
class Ascolto:
    """Ciò che l'utente ha detto, secondo il riconoscitore."""

    testo: str
    lingua: str = ""
    #: Da 0 a 1. Sotto una certa soglia conviene mostrare il testo e chiedere
    #: conferma invece di mandarlo: una domanda capita male produce una
    #: risposta giusta a una domanda che nessuno ha fatto.
    fiducia: float = 0.0
    durata: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "testo": self.testo,
            "lingua": self.lingua,
            "fiducia": round(self.fiducia, 3),
            "durata": round(self.durata, 3),
        }


@runtime_checkable
class TTSProvider(Protocol):
    """Trasforma testo in parlato."""

    name: str

    async def voci(self) -> Sequence[Dict[str, Any]]:
        """Le voci disponibili: identificativo, lingua, descrizione."""
        ...

    async def sintetizza(
        self, testo: str, *, voce: str = "", lingua: str = "it",
    ) -> Sintesi:
        ...


@runtime_checkable
class STTProvider(Protocol):
    """Trasforma parlato in testo. È il ripiego del riconoscimento del
    browser, non la via principale: mandare l'audio al server costa banda,
    latenza e un file vocale in più da custodire."""

    name: str

    async def ascolta(
        self, audio: bytes, *, media_type: str = "", lingua: str = "it",
    ) -> Ascolto:
        ...
