"""Da audio e video a testo.

I diciassette decoder coprono i documenti; il parlato no. Qui si estrae la
traccia audio con ffmpeg e la si trascrive con Whisper.

**Il parlato non è lo scritto, e va marcato come tale.** Come qualcuno parla —
esitazioni, ripetizioni, frasi costruite mentre le pensa — è materiale diverso
da come scrive: prezioso per il lessico, fuorviante se mescolato senza
distinzione. Un adapter addestrato su trascrizioni non marcate impara a
scrivere come si parla, ed è un difetto che si nota solo leggendo l'output
finale, quando è tardi.

**I timestamp sopravvivono al chunking.** Una citazione tratta da un'ora di
registrazione deve poter dire *dove*: senza, la fonte è verificabile in teoria
e irraggiungibile in pratica.

**Una trascrizione cattiva è peggio di nessuna trascrizione**, e — questo va
detto chiaramente — *nessuna difesa automatica la riconosce*.

Misurato su una registrazione volutamente difficile: Whisper ha prodotto «La
vita non è brieva, ma non è la rendima tale» per «La vita non è breve, ma noi
la rendiamo tale». La fiducia di ogni segmento era 0,60 — nella norma. Il
classificatore della digestione, a valle, le ha dato qualità 0,95 e ne ha
scritto una sintesi sensata, *inventando* un significato per «la rendima
tale». È ciò che i modelli linguistici fanno per natura: leggono attraverso
gli errori e ricostruiscono un senso plausibile.

Quindi le difese qui servono a un caso solo, e lo coprono bene: **il silenzio
trascritto come parole**, dove `no_speech_prob` è alto e il testo è
invenzione pura. Contro un errore *plausibile* non proteggono, e pretendere il
contrario sarebbe peggio che ammetterlo.

Cosa protegge davvero, in ordine:

1. **un modello grande.** `large-v3` sbaglia molto meno di `small`, e la
   trascrizione si fa una volta sola: risparmiare qui significa portarsi gli
   errori dentro il corpus per sempre;
2. **audio pulito.** Nessun modello recupera ciò che non è stato registrato;
3. **un orecchio umano su un campione.** `fiducia_media` serve a decidere
   quali file ascoltare per primi, non a dare un via libera.
"""
from __future__ import annotations

import logging
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Estensioni che passano da qui invece che dai decoder di testo.
ESTENSIONI_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
ESTENSIONI_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}

#: Sotto questa probabilità media un segmento non entra nel corpus.
#:
#: Whisper non tace mai: su un audio disturbato produce testo verosimile e
#: sbagliato. Una frase inventata dentro un corpus è peggio di una mancante,
#: perché non si distingue dalle altre e verrà citata come le altre.
SOGLIA_FIDUCIA = 0.5

#: Sotto questa fiducia media conviene ascoltare il file prima di fidarsi.
#:
#: Non scarta nulla: segnala. Scartare automaticamente una registrazione
#: intera per un valore medio significherebbe buttare via un'ora di parlato
#: per qualche minuto disturbato.
SOGLIA_ATTENZIONE = 0.7

#: Il modello predefinito. `large-v3` perché la differenza sulle lingue
#: diverse dall'inglese è netta, e la trascrizione si fa una volta sola: su
#: una GPU va molte volte più veloce del tempo reale, e risparmiare qui
#: significa portarsi errori dentro il corpus per sempre.
MODELLO_PREDEFINITO = "large-v3"

#: Whisper vuole 16 kHz mono: è il formato su cui è stato addestrato, e
#: darglielo già così evita una conversione interna a ogni chiamata.
FREQUENZA = 16_000


class TrascrizioneNonDisponibile(RuntimeError):
    """Manca uno degli strumenti necessari."""


@dataclass
class Segmento:
    """Un pezzo di parlato, con il suo posto nel tempo."""

    inizio: float
    fine: float
    testo: str
    #: Probabilità media che Whisper assegna alle parole. Bassa significa
    #: «ho sentito male», e va distinta da «non ho sentito nulla».
    fiducia: float = 1.0

    @property
    def affidabile(self) -> bool:
        return self.fiducia >= SOGLIA_FIDUCIA

    @property
    def marca(self) -> str:
        """`[12:34]`, da mettere nel testo perché sopravviva al chunking."""
        minuti, secondi = divmod(int(self.inizio), 60)
        ore, minuti = divmod(minuti, 60)
        if ore:
            return f"[{ore}:{minuti:02d}:{secondi:02d}]"
        return f"[{minuti}:{secondi:02d}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inizio": round(self.inizio, 2),
            "fine": round(self.fine, 2),
            "testo": self.testo,
            "fiducia": round(self.fiducia, 3),
        }


@dataclass
class Trascrizione:
    testo: str = ""
    segmenti: List[Segmento] = field(default_factory=list)
    lingua: str = ""
    durata: float = 0.0
    scartati: int = 0
    modello: str = ""

    @property
    def parole(self) -> int:
        return len(self.testo.split())

    @property
    def fiducia_media(self) -> float:
        """Quanto il modello credeva di aver sentito bene, in media.

        Da leggere per quello che è: un valore basso segnala quasi certamente
        una trascrizione da rivedere, ma uno alto **non garantisce nulla** —
        sul caso misurato nel docstring del modulo valeva 0,60 su un testo
        pieno di parole inesistenti. Serve a mettere in coda i file da
        ascoltare, non ad approvarli.
        """
        if not self.segmenti:
            return 0.0
        return sum(s.fiducia for s in self.segmenti) / len(self.segmenti)

    @property
    def da_riascoltare(self) -> bool:
        return self.fiducia_media < SOGLIA_ATTENZIONE

    def meta(self) -> Dict[str, Any]:
        """I metadati da attaccare al documento."""
        return {
            # Il campo che impedisce di confondere parlato e scritto a valle.
            "registro": "parlato",
            "lingua": self.lingua,
            "durata_secondi": round(self.durata, 1),
            "trascritto_con": self.modello,
            "segmenti": len(self.segmenti),
            "segmenti_scartati": self.scartati,
            "fiducia_media": round(self.fiducia_media, 3),
            "da_riascoltare": self.da_riascoltare,
        }


def e_multimediale(percorso: pathlib.Path) -> bool:
    estensione = percorso.suffix.lower()
    return estensione in ESTENSIONI_AUDIO or estensione in ESTENSIONI_VIDEO


def ffmpeg_disponibile() -> bool:
    return shutil.which("ffmpeg") is not None


def estrai_audio(sorgente: pathlib.Path, destinazione: pathlib.Path) -> None:
    """Estrae la traccia audio in WAV 16 kHz mono.

    Vale anche per un file già audio: un mp3 a 44 kHz stereo va comunque
    convertito, e farlo qui una volta è meglio che lasciarlo fare al
    trascrittore a ogni segmento.
    """
    if not ffmpeg_disponibile():
        raise TrascrizioneNonDisponibile(
            "ffmpeg non è nel PATH: serve a estrarre l'audio dai file "
            "multimediali. Su Windows si installa con `scoop install ffmpeg`."
        )

    comando = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(sorgente),
        "-vn",                      # via il video: qui interessa solo l'audio
        "-ac", "1",                 # mono
        "-ar", str(FREQUENZA),
        "-c:a", "pcm_s16le",
        str(destinazione),
    ]

    esito = subprocess.run(
        comando, capture_output=True, text=True, check=False,
    )
    if esito.returncode != 0:
        # L'ultima riga di ffmpeg è quella che dice cosa è andato storto; le
        # precedenti sono la configurazione della build.
        ultima = (esito.stderr or "").strip().splitlines()[-1:] or ["errore ignoto"]
        raise TrascrizioneNonDisponibile(
            f"ffmpeg non è riuscito a leggere {sorgente.name}: {ultima[0]}"
        )


class Trascrittore:
    """Whisper, caricato una volta sola.

    Il modello pesa gigabyte e impiega decine di secondi a caricarsi:
    ricrearlo a ogni file trasformerebbe una coda di cento registrazioni in
    un'ora di soli caricamenti.
    """

    def __init__(
        self,
        modello: str = MODELLO_PREDEFINITO,
        *,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self.modello = modello
        self._device = device
        self._compute_type = compute_type
        self._whisper = None

    def _carica(self):
        if self._whisper is not None:
            return self._whisper

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TrascrizioneNonDisponibile(
                "faster-whisper non è installato: `pip install faster-whisper`"
            ) from exc

        device, compute = self._device, self._compute_type
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if compute == "auto":
            # `float16` su GPU, `int8` su CPU: su CPU la precisione piena
            # rende la trascrizione più lenta dell'ascolto, e a quel punto
            # nessuno la esegue.
            compute = "float16" if device == "cuda" else "int8"

        logger.info(
            "Carico Whisper %s su %s (%s)", self.modello, device, compute,
        )
        self._whisper = WhisperModel(
            self.modello, device=device, compute_type=compute,
        )
        return self._whisper

    def trascrivi(
        self,
        percorso: pathlib.Path,
        *,
        lingua: Optional[str] = None,
        soglia: float = SOGLIA_FIDUCIA,
    ) -> Trascrizione:
        """Trascrive un file audio o video."""
        percorso = pathlib.Path(percorso)
        if not percorso.exists():
            raise FileNotFoundError(percorso)

        with tempfile.TemporaryDirectory(prefix="persona-audio-") as tmp:
            wav = pathlib.Path(tmp) / "audio.wav"
            estrai_audio(percorso, wav)
            return self._trascrivi_wav(wav, lingua=lingua, soglia=soglia)

    def _trascrivi_wav(
        self, wav: pathlib.Path, *, lingua: Optional[str], soglia: float,
    ) -> Trascrizione:
        modello = self._carica()

        segmenti_grezzi, info = modello.transcribe(
            str(wav),
            language=lingua,
            # Rilevamento del parlato: salta i silenzi invece di trascriverli,
            # ed è ciò che impedisce a Whisper di inventare parole nel vuoto —
            # il suo modo tipico di fallire su registrazioni con lunghe pause.
            vad_filter=True,
            beam_size=5,
        )

        trascrizione = Trascrizione(
            lingua=info.language or (lingua or ""),
            durata=float(getattr(info, "duration", 0.0) or 0.0),
            modello=self.modello,
        )

        pezzi: List[str] = []
        for grezzo in segmenti_grezzi:
            segmento = Segmento(
                inizio=float(grezzo.start),
                fine=float(grezzo.end),
                testo=(grezzo.text or "").strip(),
                fiducia=_fiducia(grezzo),
            )
            if not segmento.testo:
                continue

            if segmento.fiducia < soglia:
                trascrizione.scartati += 1
                logger.debug(
                    "Segmento %s scartato (fiducia %.2f): %r",
                    segmento.marca, segmento.fiducia, segmento.testo[:60],
                )
                continue

            trascrizione.segmenti.append(segmento)
            pezzi.append(f"{segmento.marca} {segmento.testo}")

        trascrizione.testo = "\n\n".join(pezzi)

        if trascrizione.scartati:
            logger.info(
                "Trascrizione: %d segmenti tenuti, %d scartati per fiducia bassa",
                len(trascrizione.segmenti), trascrizione.scartati,
            )
        return trascrizione


def _fiducia(segmento: Any) -> float:
    """La probabilità che Whisper assegna a un segmento.

    `avg_logprob` è il logaritmo della probabilità media: si riporta in [0,1]
    con l'esponenziale. Valori sotto −1 corrispondono a un modello che ha
    sentito poco e indovinato molto.
    """
    import math

    logprob = getattr(segmento, "avg_logprob", None)
    if logprob is None:
        return 1.0

    # `no_speech_prob` alta significa che lì probabilmente non parlava
    # nessuno: il testo prodotto è rumore interpretato come parole.
    silenzio = float(getattr(segmento, "no_speech_prob", 0.0) or 0.0)
    if silenzio > 0.6:
        return 0.0

    return max(0.0, min(1.0, math.exp(float(logprob))))
