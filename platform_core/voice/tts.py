"""I fornitori di sintesi.

Due, per ragioni diverse.

`SintesiOpenAICompatibile` parla con qualunque servizio che esponga
`/v1/audio/speech` — OpenAI, LM Studio con un modello di sintesi, Kokoro,
un'istanza propria. Non restituisce tempi: quelli si misurano dopo, con
l'allineamento.

`SintesiMuta` non produce suono. Esiste perché l'intero percorso — diritto
alla voce, generazione, allineamento, visemi, consegna al client — deve
essere provabile senza una chiave e senza rete, e perché su un'installazione
senza fornitore configurato l'endpoint deve poter rispondere qualcosa di
onesto invece di un errore che sembra un guasto.
"""
from __future__ import annotations

import logging
import math
import struct
from typing import Any, Dict, List, Optional, Sequence

import httpx

from ..settings import Settings, get_settings
from .base import Sintesi, SintesiNonDisponibile
from .visemi import Parola

logger = logging.getLogger(__name__)

#: Quanto tempo dare a una sintesi.
#:
#: Più generoso di una richiesta normale perché la voce si genera tutta prima
#: di poterla mandare — non c'è streaming utile quando poi va allineata — e su
#: una risposta lunga un modello locale ci mette decine di secondi.
ATTESA = 120.0


class SintesiOpenAICompatibile:
    """`/v1/audio/speech`, con la voce e il modello scelti da chi chiama."""

    name = "openai-compatibile"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        modello: str = "",
        voce_predefinita: str = "",
        settings: Optional[Settings] = None,
    ) -> None:
        settings = settings or get_settings()
        self._base_url = (base_url or settings.tts_base_url).rstrip("/")
        self._api_key = api_key or settings.tts_api_key
        self._modello = modello or settings.tts_model
        self._voce = voce_predefinita or settings.tts_voice

    async def voci(self) -> Sequence[Dict[str, Any]]:
        """Le voci che il servizio dichiara.

        Molti servizi compatibili non espongono `/v1/audio/voices`: un elenco
        vuoto significa «non lo so», e chi amministra deve poter scrivere il
        nome della voce a mano invece di trovarsi un menu vuoto e nessuna
        spiegazione.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                risposta = await client.get(
                    f"{self._base_url}/audio/voices",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            if risposta.status_code != 200:
                return []
            corpo = risposta.json()
        except (httpx.HTTPError, ValueError):
            logger.info("Il fornitore di voce non elenca le sue voci")
            return []

        voci = corpo.get("voices") if isinstance(corpo, dict) else corpo
        if not isinstance(voci, list):
            return []
        return [
            v if isinstance(v, dict) else {"id": str(v), "lingua": ""}
            for v in voci
        ]

    async def sintetizza(
        self, testo: str, *, voce: str = "", lingua: str = "it",
    ) -> Sintesi:
        if not testo.strip():
            raise ValueError("non si sintetizza il silenzio")

        corpo = {
            "model": self._modello,
            "input": testo,
            "voice": voce or self._voce,
            "response_format": "mp3",
        }

        try:
            async with httpx.AsyncClient(timeout=ATTESA) as client:
                risposta = await client.post(
                    f"{self._base_url}/audio/speech",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=corpo,
                )
        except httpx.HTTPError as exc:
            raise SintesiNonDisponibile(
                f"il fornitore di voce non risponde: {exc}"
            ) from exc

        if risposta.status_code != 200:
            raise SintesiNonDisponibile(
                f"il fornitore di voce ha rifiutato la richiesta "
                f"({risposta.status_code}): {risposta.text[:200]}"
            )

        return Sintesi(
            audio=risposta.content,
            media_type=risposta.headers.get("content-type", "audio/mpeg"),
            voce=voce or self._voce,
            modello=self._modello,
        )


class SintesiMuta:
    """Silenzio della durata giusta, per provare il percorso senza fornitore.

    La durata non è un numero a caso: è stimata dalla lunghezza del testo a
    una velocità di lettura normale, così che il client riceva qualcosa di
    plausibile da animare e che un difetto nella catena — tempi assenti,
    visemi che eccedono l'audio — si veda qui invece che in produzione.

    I tempi delle parole li **dichiara stimati**, e questo è il punto: chi li
    riceve sa di non poterli usare per un labiale serio.
    """

    name = "muta"

    #: Sillabe al secondo di un parlato normale in italiano.
    VELOCITA = 5.5

    def __init__(self, *, frequenza: int = 22050) -> None:
        self._frequenza = frequenza

    async def voci(self) -> Sequence[Dict[str, Any]]:
        return [{"id": "muta", "lingua": "it", "nome": "Silenzio (sviluppo)"}]

    async def sintetizza(
        self, testo: str, *, voce: str = "", lingua: str = "it",
    ) -> Sintesi:
        if not testo.strip():
            raise ValueError("non si sintetizza il silenzio")

        parole_testo = testo.split()
        sillabe = sum(max(1, _sillabe(p)) for p in parole_testo)
        durata = max(0.5, sillabe / self.VELOCITA)

        parole: List[Parola] = []
        cursore = 0.0
        for parola in parole_testo:
            quanto = max(0.12, _sillabe(parola) / self.VELOCITA)
            parole.append(Parola(parola, cursore, cursore + quanto))
            cursore += quanto

        return Sintesi(
            audio=_wav_silenzioso(durata, self._frequenza),
            media_type="audio/wav",
            durata=durata,
            voce="muta",
            modello="nessuno",
            parole=parole,
            origine_tempi="stima",
        )


def _sillabe(parola: str) -> int:
    """Quante sillabe, contando i gruppi di vocali.

    Approssimazione grossolana, e va bene: serve solo a dare al silenzio una
    durata verosimile. In italiano i dittonghi la fanno sbagliare per difetto,
    che è l'errore innocuo — il silenzio finisce un po' prima.
    """
    gruppi = 0
    dentro = False
    for lettera in parola.lower():
        vocale = lettera in "aeiouàèéìòù"
        if vocale and not dentro:
            gruppi += 1
        dentro = vocale
    return gruppi


def _wav_silenzioso(durata: float, frequenza: int) -> bytes:
    """Un WAV valido e muto.

    Scritto a mano invece che con `wave`: sono quaranta byte di intestazione,
    e la dipendenza da un file temporaneo per produrre silenzio sarebbe più
    codice di questo.
    """
    campioni = max(1, int(durata * frequenza))
    dati = b"\x00\x00" * campioni
    return b"".join([
        b"RIFF",
        struct.pack("<I", 36 + len(dati)),
        b"WAVEfmt ",
        struct.pack("<IHHIIHH", 16, 1, 1, frequenza, frequenza * 2, 2, 16),
        b"data",
        struct.pack("<I", len(dati)),
        dati,
    ])
