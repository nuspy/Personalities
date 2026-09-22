"""I tempi delle parole, misurati sull'audio.

Quasi nessun fornitore di sintesi li restituisce. L'alternativa ovvia è
stimarli — tante lettere, tanti millisecondi — e produce un labiale che a
inizio frase sembra giusto e a metà è mezzo secondo avanti: l'errore si
accumula, e l'occhio nota una bocca fuori sincrono molto prima di notare una
forma sbagliata.

La misura vera costa una trascrizione dell'audio appena sintetizzato. Sembra
un giro assurdo — si fa dire alla macchina ciò che le si è appena fatto dire —
ed è invece il modo standard di ottenere un allineamento forzato: Whisper
restituisce i tempi di ogni parola, e qui il testo di riferimento è noto, così
che un errore di riconoscimento si possa correggere invece di propagarsi.

**Il testo che vale è quello originale, non quello riconosciuto.** Whisper
può sentire «perché» dove era scritto «per che»: si tengono i suoi *tempi* e
le parole di partenza, allineandoli per posizione. Se i conteggi non
corrispondono, l'allineamento si dichiara fallito invece di accoppiare parole
a caso — meglio nessun labiale che uno che si muove sulle parole sbagliate.
"""
from __future__ import annotations

import logging
import pathlib
import re
import tempfile
from typing import Any, List, Optional, Sequence

from .visemi import Parola

logger = logging.getLogger(__name__)

#: Modello per l'allineamento, non per la trascrizione.
#:
#: Piccolo apposta: qui il testo è già noto e servono solo i tempi, che
#: `small` misura quanto `large-v3` a un quinto del costo. Usare il modello
#: grande spenderebbe secondi di GPU per migliorare parole che verranno
#: comunque buttate via.
MODELLO_ALLINEAMENTO = "small"


class AllineamentoNonDisponibile(RuntimeError):
    """Manca ciò che serve per misurare i tempi."""


def _parole_attese(testo: str) -> List[str]:
    """Le parole del testo di partenza, nella forma che si vuole animare."""
    return [p for p in re.findall(r"[^\s]+", testo) if re.search(r"\w", p)]


class Allineatore:
    """Misura i tempi delle parole di un audio dal testo noto.

    Il modello si carica una volta sola: pesa centinaia di megabyte e decine
    di secondi, e ricrearlo a ogni risposta renderebbe la voce inutilizzabile
    proprio nel caso che conta — una conversazione.
    """

    def __init__(self, modello: str = MODELLO_ALLINEAMENTO, *, device: str = "auto"):
        self.modello = modello
        self._device = device
        self._whisper = None

    def disponibile(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def _carica(self):
        if self._whisper is not None:
            return self._whisper

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AllineamentoNonDisponibile(
                "faster-whisper non è installato: i tempi delle parole non si "
                "possono misurare, e senza di essi il labiale sarebbe animato "
                "su una stima."
            ) from exc

        device = self._device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info("Carico Whisper %s per l'allineamento (%s)", self.modello, device)
        self._whisper = WhisperModel(
            self.modello,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        return self._whisper

    def allinea(
        self, audio: bytes, testo: str, *, lingua: str = "it",
    ) -> List[Parola]:
        """I tempi delle parole di `testo` dentro `audio`.

        Lista vuota quando l'allineamento non è affidabile: è un esito
        legittimo, e chi chiama deve trattarlo come «nessun labiale» invece di
        inventarne uno.
        """
        attese = _parole_attese(testo)
        if not attese or not audio:
            return []

        modello = self._carica()

        with tempfile.TemporaryDirectory(prefix="persona-voce-") as tmp:
            percorso = pathlib.Path(tmp) / "sintesi.audio"
            percorso.write_bytes(audio)

            segmenti, _info = modello.transcribe(
                str(percorso),
                language=lingua or None,
                word_timestamps=True,
                # Nessun filtro del parlato: l'audio è una sintesi, non una
                # registrazione. Saltare i «silenzi» qui toglierebbe le pause
                # fra le frasi, che sono proprio i momenti in cui la bocca
                # deve tornare a riposo.
                vad_filter=False,
                beam_size=1,
            )

            misurate = [
                (p.word.strip(), float(p.start), float(p.end))
                for segmento in segmenti
                for p in (getattr(segmento, "words", None) or [])
                if p.word and p.word.strip()
            ]

        return _accoppia(attese, misurate)


def _accoppia(
    attese: Sequence[str], misurate: Sequence[tuple],
) -> List[Parola]:
    """Unisce le parole di partenza ai tempi misurati.

    Per posizione, e solo se i conteggi corrispondono. Un allineamento
    approssimato — accoppiare le prime N e scartare il resto — produrrebbe una
    bocca che si muove sulle parole sbagliate per tutta la seconda metà della
    frase, che è il difetto peggiore possibile perché sembra un problema del
    modello 3D.
    """
    if not misurate:
        return []

    if len(attese) != len(misurate):
        logger.info(
            "Allineamento scartato: %d parole attese, %d misurate",
            len(attese), len(misurate),
        )
        return []

    return [
        Parola(testo=attesa, inizio=inizio, fine=fine)
        for attesa, (_riconosciuta, inizio, fine) in zip(attese, misurate)
    ]
