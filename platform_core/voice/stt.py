"""Il riconoscimento vocale lato server.

**È il ripiego, non la via principale.** Il riconoscimento del browser
(`SpeechRecognition`) gira sulla macchina di chi parla, non costa banda, non
manda un file vocale da nessuna parte e restituisce risultati parziali mentre
si parla — tre vantaggi che nessun servizio remoto può pareggiare. Esiste però
solo su alcuni browser, e su quelli che non ce l'hanno la dettatura
semplicemente non funzionerebbe.

Quindi questo: stesso Whisper dell'ingestione, una registrazione alla volta.

**L'audio non si conserva.** Arriva, viene trascritto, e il file sparisce con
la directory temporanea. Una registrazione della voce di qualcuno è il genere
di dato che non si tiene «per ora», perché poi resta: la trascrizione è ciò
che serve alla conversazione, l'audio no.

**La fiducia si riporta, non si nasconde.** Whisper produce testo verosimile
anche su registrazioni pessime: una frase inventata con sicurezza è peggio di
nessuna frase, perché il modello risponde a una domanda che nessuno ha fatto.
Con fiducia bassa chi chiama deve mostrare il testo e far confermare invece di
mandarlo.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import tempfile
from typing import Optional

from ..knowledge.transcription import (
    Trascrittore, TrascrizioneNonDisponibile, estrai_audio,
)
from .base import Ascolto

logger = logging.getLogger(__name__)

#: Il modello per la dettatura.
#:
#: Più grande di quello dell'allineamento e più piccolo di quello
#: dell'ingestione: qui il testo non è noto — va riconosciuto davvero — ma si
#: tratta di una frase, non di un'ora di registrazione, e l'attesa la sta
#: subendo qualcuno che guarda lo schermo.
MODELLO_DETTATURA = "medium"

#: Oltre questa durata non è più dettatura.
#:
#: Una domanda si fa in qualche secondo. Un file di dieci minuti è un caricamento
#: travestito da dettatura, e va per l'ingestione — dove c'è una coda, un
#: avanzamento e nessuno che aspetti guardando.
SECONDI_MASSIMI = 120


class AscoltoWhisper:
    """Dettatura con Whisper, sul nodo che ha l'acceleratore."""

    name = "whisper"

    def __init__(self, modello: str = MODELLO_DETTATURA) -> None:
        self._trascrittore = Trascrittore(modello)

    def disponibile(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    async def ascolta(
        self, audio: bytes, *, media_type: str = "", lingua: str = "it",
    ) -> Ascolto:
        if not audio:
            raise ValueError("nessun audio da trascrivere")

        # In un thread: `transcribe` è sincrona e occupa la CPU o la GPU per
        # secondi. Nel loop fermerebbe ogni altra richiesta del processo,
        # comprese quelle di chi non sta dettando niente.
        return await asyncio.get_running_loop().run_in_executor(
            None, self._trascrivi, audio, media_type, lingua,
        )

    def _trascrivi(self, audio: bytes, media_type: str, lingua: str) -> Ascolto:
        with tempfile.TemporaryDirectory(prefix="persona-dettatura-") as tmp:
            cartella = pathlib.Path(tmp)
            grezzo = cartella / f"registrazione{_estensione(media_type)}"
            grezzo.write_bytes(audio)

            # Passa da ffmpeg anche quando sembra già un WAV: il browser
            # registra in webm/opus con parametri che variano, e Whisper vuole
            # 16 kHz mono. Fidarsi del tipo dichiarato dal client significa
            # scoprire il problema come una trascrizione vuota.
            wav = cartella / "audio.wav"
            estrai_audio(grezzo, wav)

            trascrizione = self._trascrittore.trascrivi(wav, lingua=lingua)

        if trascrizione.durata > SECONDI_MASSIMI:
            logger.info(
                "Dettatura di %.0f secondi: oltre il limite di %d",
                trascrizione.durata, SECONDI_MASSIMI,
            )

        return Ascolto(
            testo=trascrizione.testo.strip(),
            lingua=trascrizione.lingua,
            fiducia=trascrizione.fiducia_media,
            durata=trascrizione.durata,
        )


def _estensione(media_type: str) -> str:
    """Un'estensione plausibile per ffmpeg.

    Non serve che sia giusta — ffmpeg riconosce il formato dal contenuto — ma
    un'estensione coerente evita che si insospettisca su file senza nome.
    """
    return {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/flac": ".flac",
    }.get(media_type.split(";")[0].strip().lower(), ".bin")
