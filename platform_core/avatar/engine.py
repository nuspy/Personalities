"""Gli avatar: cosa il client deve disegnare.

**Il server non genera mai fotogrammi.** Descrive un avatar e consegna i
tempi; a disegnare è il browser. Un volto parlante generato lato server
costerebbe rendering e banda video per ogni ascoltatore, non si potrebbe
riusare per altro, e legherebbe la qualità dell'animazione alla latenza della
rete invece che alla macchina di chi guarda.

Quindi un `AvatarEngine` non anima: **valida** una configurazione e produce il
*descrittore* che il client sa rendere. Tre implementazioni, in ordine di
costo crescente per chi costruisce la personalità:

| | cosa serve | cosa fa la bocca |
|---|---|---|
| `immagine` | un'immagine | niente: si anima l'onda sonora accanto |
| `video` | clip in ciclo per stato | cambia clip fra fermo e parlante |
| `modello` | un `.glb` con morph target | segue i visemi, forma per forma |

**Il labiale è una facoltà, non un requisito.** Un avatar a immagine ferma è
una scelta legittima — molte voci storiche hanno un solo ritratto — e
pretendere un modello 3D per poter parlare significherebbe che la voce
funziona solo per chi ha commissionato una testa. Il descrittore dichiara
`labiale` perché il client sappia se ha senso chiedere i visemi: chiederli
costa un allineamento dell'audio, e spenderlo per un'immagine ferma è lavoro
buttato.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..voice.visemi import VISEMI

logger = logging.getLogger(__name__)

#: I tipi di avatar che il client sa rendere.
TIPI_AVATAR = ("immagine", "video", "modello")


class ConfigurazioneAvatarNonValida(ValueError):
    """La configurazione non descrive qualcosa di disegnabile."""


@dataclass
class Descrittore:
    """Cosa il client riceve per poter disegnare."""

    tipo: str
    #: Dove sta la risorsa principale: l'immagine, il video di riposo, il
    #: modello.
    uri: str
    #: Vero se questo avatar sa muovere la bocca sui visemi. Il client lo
    #: guarda **prima** di chiedere la sintesi con i tempi: misurarli costa
    #: una trascrizione dell'audio, e per un ritratto fermo è sprecata.
    labiale: bool = False
    #: Le pose disponibili, per un modello: nome del visema → nome del morph
    #: target nel file. Senza questa corrispondenza il client avrebbe i tempi
    #: e non saprebbe quale forma muovere.
    pose: Dict[str, str] = field(default_factory=dict)
    #: Risorse aggiuntive per tipo: le clip di un video, lo sfondo, la scala.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tipo": self.tipo,
            "uri": self.uri,
            "labiale": self.labiale,
            "pose": self.pose,
            "extra": self.extra,
        }


@runtime_checkable
class AvatarEngine(Protocol):
    """Valida una configurazione e la traduce in un descrittore."""

    tipo: str

    def valida(self, config: Dict[str, Any]) -> None:
        """Solleva `ConfigurazioneAvatarNonValida` se manca qualcosa.

        Al salvataggio e non al disegno: un avatar rotto scoperto mentre
        qualcuno conversa è un volto che non compare e un errore nella console
        del browser, cioè nessuna informazione per chi l'ha configurato.
        """
        ...

    def descrivi(self, config: Dict[str, Any]) -> Descrittore:
        ...


class ImmagineFerma:
    """Un ritratto. Nessun labiale, e va bene così."""

    tipo = "immagine"

    def valida(self, config: Dict[str, Any]) -> None:
        if not (config or {}).get("uri"):
            raise ConfigurazioneAvatarNonValida(
                "serve l'indirizzo dell'immagine (`uri`)"
            )

    def descrivi(self, config: Dict[str, Any]) -> Descrittore:
        self.valida(config)
        return Descrittore(
            tipo=self.tipo,
            uri=config["uri"],
            labiale=False,
            # Mentre la voce parla il client anima l'onda sonora accanto al
            # ritratto: dice «sta parlando» senza fingere un labiale che
            # un'immagine ferma non può avere.
            extra={"onda": bool(config.get("onda", True))},
        )


class VideoInCiclo:
    """Clip in ciclo, una per stato.

    Il labiale non segue i visemi: si alterna fra fermo e parlante. È un
    compromesso onesto — la bocca si muove nei momenti giusti senza formare le
    parole giuste — e costa una ripresa invece di una modellazione.
    """

    tipo = "video"

    #: Gli stati che il client sa incrociare. `fermo` è obbligatorio: è quello
    #: che si vede quando non succede nulla, cioè quasi sempre.
    STATI = ("fermo", "parlante", "ascolto")

    def valida(self, config: Dict[str, Any]) -> None:
        clip = (config or {}).get("clip") or {}
        if not isinstance(clip, dict) or not clip.get("fermo"):
            raise ConfigurazioneAvatarNonValida(
                "serve almeno la clip `fermo`: è quella che si vede quando "
                "non succede nulla"
            )

        ignoti = set(clip) - set(self.STATI)
        if ignoti:
            raise ConfigurazioneAvatarNonValida(
                f"stati sconosciuti: {sorted(ignoti)}; "
                f"quelli previsti sono {list(self.STATI)}"
            )

    def descrivi(self, config: Dict[str, Any]) -> Descrittore:
        self.valida(config)
        clip = config["clip"]
        return Descrittore(
            tipo=self.tipo,
            uri=clip["fermo"],
            labiale=False,
            extra={
                "clip": clip,
                # Quanto dura la dissolvenza fra una clip e l'altra. Un taglio
                # netto si vede come uno scatto proprio nel momento in cui
                # l'attenzione è sul volto.
                "dissolvenza": float(config.get("dissolvenza", 0.15)),
            },
        )


class Modello3D:
    """Un `.glb` con i morph target delle forme della bocca.

    L'unico che segue i visemi davvero. In cambio pretende che il file porti
    le pose e che qualcuno dica come si chiamano: i nomi variano fra strumenti
    — `viseme_PP`, `mouthPucker`, `A`, `01_Ah` — e indovinarli significa un
    volto immobile senza alcun errore da nessuna parte.
    """

    tipo = "modello"

    def valida(self, config: Dict[str, Any]) -> None:
        config = config or {}
        if not config.get("uri"):
            raise ConfigurazioneAvatarNonValida("serve l'indirizzo del modello (`uri`)")

        pose = config.get("pose") or {}
        if not isinstance(pose, dict) or not pose:
            raise ConfigurazioneAvatarNonValida(
                "servono le pose: la corrispondenza fra i visemi e i morph "
                "target del file. Senza, il client ha i tempi e non sa quale "
                "forma muovere."
            )

        ignoti = set(pose) - set(VISEMI)
        if ignoti:
            raise ConfigurazioneAvatarNonValida(
                f"visemi sconosciuti nelle pose: {sorted(ignoti)}; "
                f"quelli previsti sono {list(VISEMI)}"
            )

        if "X" not in pose:
            raise ConfigurazioneAvatarNonValida(
                "manca la posa `X`, la bocca a riposo: senza, una pausa "
                "lascerebbe il volto congelato nell'ultima forma pronunciata"
            )

    def descrivi(self, config: Dict[str, Any]) -> Descrittore:
        self.valida(config)
        mancanti = sorted(set(VISEMI) - set(config["pose"]))
        if mancanti:
            # Non un errore: un modello con dieci pose su quindici anima
            # peggio, non male. Le forme mancanti il client le rende con la
            # più vicina, e chi ha configurato l'avatar deve sapere quali
            # sono invece di scoprirlo guardando.
            logger.info("Avatar 3D senza le pose %s", mancanti)

        return Descrittore(
            tipo=self.tipo,
            uri=config["uri"],
            labiale=True,
            pose=dict(config["pose"]),
            extra={
                "scala": float(config.get("scala", 1.0)),
                "camera": config.get("camera") or {},
                "pose_mancanti": mancanti,
            },
        )


_MOTORI: Dict[str, AvatarEngine] = {
    ImmagineFerma.tipo: ImmagineFerma(),
    VideoInCiclo.tipo: VideoInCiclo(),
    Modello3D.tipo: Modello3D(),
}


def motore_per(tipo: str) -> AvatarEngine:
    motore = _MOTORI.get(tipo)
    if motore is None:
        raise ConfigurazioneAvatarNonValida(
            f"tipo di avatar sconosciuto: «{tipo}»; "
            f"quelli previsti sono {list(TIPI_AVATAR)}"
        )
    return motore


def descrivi(tipo: str, config: Optional[Dict[str, Any]]) -> Descrittore:
    return motore_per(tipo).descrivi(config or {})


def valida(tipo: str, config: Optional[Dict[str, Any]]) -> None:
    motore_per(tipo).valida(config or {})
