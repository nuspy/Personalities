"""Endpoint delle capacita' della piattaforma.

Il frontend e la console di amministrazione si configurano da qui: quali
modalita' offrire per una personalita', quali pulsanti abilitare, cosa
scrivere accanto a quelli che restano spenti. Nessuna ipotesi cablata da
nessuna parte, perche' la risposta cambia a seconda di dove gira il servizio.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ...capabilities.registry import CapabilityRegistry, Feature
from ...settings import Settings, get_settings
from ..deps import get_capability_registry

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities")
def read_capabilities(
    registry: CapabilityRegistry = Depends(get_capability_registry),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Cosa la piattaforma sa fare ora, e perche' non fa il resto."""
    capabilities = registry.platform_capabilities(
        local_engines=_local_engines(settings), voice=_voce(settings),
    )
    return {
        "environment": settings.environment,
        **capabilities.as_dict(),
    }


def require_feature(feature: Feature):
    """Dipendenza che nega l'accesso a un endpoint quando la funzione manca.

    Il controllo nell'interfaccia e' cortesia verso chi guarda; questo e' la
    regola. Risponde **409 Conflict**, non 500: la richiesta e' legittima e
    lo stato del servizio e' sano — semplicemente non c'e' nessuno che possa
    eseguirla, e il messaggio dice cosa fare.
    """

    def dependency(
        registry: CapabilityRegistry = Depends(get_capability_registry),
        settings: Settings = Depends(get_settings),
    ) -> None:
        available, reason = registry.can(
            feature,
            local_engines=_local_engines(settings),
            voice=_voce(settings),
        )
        if not available:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "feature_unavailable",
                    "feature": feature.value,
                    "label": feature.label,
                    "reason": reason,
                },
            )

    return dependency


def _local_engines(settings: Settings) -> Optional[list]:
    """Motori locali configurati.

    Provengono dalla configurazione dei provider, non dai worker: un motore
    di inferenza e' un servizio raggiungibile in rete. Sara' `llmswitch` a
    fornirli quando il registro provider entrera' in funzione; fino ad allora
    valgono quelli dichiarati nell'ambiente.
    """
    return list(settings.local_engine_urls)


def _voce(settings: Settings) -> dict:
    """Cosa la piattaforma sa fare con la voce, adesso.

    Rilevato e non dedotto dalle impostazioni soltanto: `tts_align_words`
    acceso con faster-whisper assente darebbe un labiale annunciato e mai
    consegnato, cioe' un avatar che secondo l'interfaccia dovrebbe muovere la
    bocca e non la muove — il difetto che sembra un guasto del modello 3D.
    """
    from ..deps import get_allineatore

    return {
        Feature.VOICE_OUTPUT: bool(settings.tts_enabled),
        Feature.LIP_SYNC: bool(settings.tts_enabled) and get_allineatore() is not None,
    }
