"""Gli avatar. Il server descrive, il client disegna."""

from .engine import (
    TIPI_AVATAR, AvatarEngine, ConfigurazioneAvatarNonValida, Descrittore,
    ImmagineFerma, Modello3D, VideoInCiclo, descrivi, motore_per, valida,
)

__all__ = [
    "AvatarEngine",
    "ConfigurazioneAvatarNonValida",
    "Descrittore",
    "ImmagineFerma",
    "Modello3D",
    "TIPI_AVATAR",
    "VideoInCiclo",
    "descrivi",
    "motore_per",
    "valida",
]
