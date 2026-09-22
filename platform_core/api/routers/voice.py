"""Voce e avatar.

**Il server manda audio e tempi, mai fotogrammi.** A disegnare è il client:
un volto parlante generato qui costerebbe rendering e banda video per ogni
ascoltatore, e legherebbe la qualità dell'animazione alla rete invece che
alla macchina di chi guarda.

**La voce è un diritto del piano.** `Diritti.voce` esisteva da prima ed era
l'unico a non essere mai consultato: qui serve. Il rifiuto è un 403 col
motivo, distinto dal 402 dei crediti, perché si risolve in un altro modo.

**La sintesi non si paga a parte.** La risposta è già stata pagata quando è
stata generata, e far pagare anche il leggerla ad alta voce significherebbe
che ascoltare costa più che leggere — una differenza che nessuno si aspetta e
che punisce chi usa la voce perché non può leggere.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from ...auth.dependencies import CurrentUser, DbSession
from ...avatar.engine import ConfigurazioneAvatarNonValida, descrivi
from ...billing.plans import GestoreAbbonamenti
from ...domain.avatar_models import Avatar
from ...domain.knowledge_models import Personality, PersonalityVersion
from ...voice.base import Sintesi, SintesiNonDisponibile, TTSProvider
from ...voice.visemi import VISEMI, visemi_da
from ..deps import get_allineatore, get_stt, get_tts

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voce"])

#: Oltre questa lunghezza la sintesi si rifiuta.
#:
#: Non un limite tecnico: una risposta di ventimila caratteri messa in voce
#: occupa il fornitore per minuti e produce un file che nessuno ascolterà
#: intero. Il taglio va fatto da chi chiede — una frase, un paragrafo — e
#: dirlo è più utile che sintetizzare venti minuti di parlato.
CARATTERI_MASSIMI = 4000


class DaLeggere(BaseModel):
    testo: str = Field(min_length=1, max_length=CARATTERI_MASSIMI)
    #: Quale voce. Vuoto: quella predefinita del fornitore, o quella della
    #: personalità se se ne indica una.
    voce: str = Field(default="", max_length=80)
    personality: Optional[str] = Field(default=None, max_length=80)
    lingua: str = Field(default="it", max_length=10)
    #: Misurare i tempi delle parole costa una trascrizione dell'audio appena
    #: prodotto. Ha senso solo se qualcosa li userà: per un ritratto fermo è
    #: lavoro buttato, e il client lo sa dal descrittore dell'avatar.
    labiale: bool = True


async def _puo_usare_la_voce(session, user) -> None:
    diritti = await GestoreAbbonamenti(session).diritti_di(user.id)
    if diritti.voce:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Il tuo accesso non comprende la voce."
            if diritti.predefiniti
            else f"Il piano «{diritti.piano}» non comprende la voce."
        ),
    )


@router.get("/voice/voices")
async def voci(
    user: CurrentUser,
    session: DbSession,
    tts: Annotated[TTSProvider, Depends(get_tts)],
) -> Dict[str, Any]:
    """Le voci disponibili, e i visemi che il client deve saper disegnare."""
    await _puo_usare_la_voce(session, user)
    return {
        "fornitore": tts.name,
        "voci": list(await tts.voci()),
        "visemi": list(VISEMI),
    }


@router.post("/voice/speak")
async def leggi(
    payload: DaLeggere,
    user: CurrentUser,
    session: DbSession,
    tts: Annotated[TTSProvider, Depends(get_tts)],
    allineatore: Annotated[Any, Depends(get_allineatore)],
) -> Dict[str, Any]:
    """Mette in voce un testo, con i tempi per animare la bocca.

    L'audio torna in base64 dentro il JSON invece che come corpo binario: i
    tempi e i visemi devono arrivare **insieme** all'audio, e due risposte
    separate significherebbero un client che anima prima di avere il suono o
    riproduce prima di avere le forme. Il sovrappeso della codifica è un
    terzo, su file di qualche decina di kilobyte.
    """
    await _puo_usare_la_voce(session, user)

    voce = payload.voce
    if not voce and payload.personality:
        voce = await _voce_di(session, payload.personality)

    try:
        sintesi: Sintesi = await tts.sintetizza(
            payload.testo, voce=voce, lingua=payload.lingua,
        )
    except SintesiNonDisponibile as exc:
        # 503 e non 500: non dipende dall'utente, e ripetere fra un minuto
        # può benissimo funzionare.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.labiale:
        _misura_i_tempi(sintesi, payload.testo, payload.lingua, allineatore)

    if sintesi.parole:
        sintesi.visemi = visemi_da(sintesi.parole, durata_totale=sintesi.durata)

    return sintesi.to_dict(con_audio=True)


def _misura_i_tempi(sintesi: Sintesi, testo: str, lingua: str, allineatore) -> None:
    """Riempie i tempi delle parole, se si possono misurare.

    Un fallimento qui non è un errore della richiesta: si consegna l'audio
    senza labiale, e `origine_tempi` resta vuoto perché il client sappia di
    non doverne animare uno. Inventare tempi darebbe una bocca che si muove
    sulle parole sbagliate — peggio di una ferma, perché sembra un difetto del
    modello 3D.
    """
    if sintesi.parole and sintesi.origine_tempi == "fornitore":
        return
    if allineatore is None:
        return

    try:
        parole = allineatore.allinea(sintesi.audio, testo, lingua=lingua)
    except Exception:  # noqa: BLE001
        logger.warning("Allineamento non riuscito: la voce va senza labiale", exc_info=True)
        return

    if parole:
        sintesi.parole = parole
        sintesi.origine_tempi = "allineamento"
        if not sintesi.durata:
            sintesi.durata = parole[-1].fine


async def _voce_di(session, slug: str) -> str:
    """Il timbro dichiarato dalla versione corrente di una personalità.

    Vuoto quando non ne dichiara uno: il fornitore usa la sua predefinita, che
    è meglio di un errore — una voce generica è comunque una voce, e rifiutare
    di leggere perché nessuno ha scelto il timbro toglierebbe la funzione a
    ogni personalità non ancora rifinita.
    """
    versione = await session.scalar(
        select(PersonalityVersion)
        .join(Personality, Personality.current_version_id == PersonalityVersion.id)
        .where(Personality.slug == slug, Personality.status == "published")
    )
    if versione is None:
        return ""
    return str((versione.voice_config or {}).get("voce") or "")


@router.get("/personalities/{slug}/avatar")
async def avatar_di(slug: str, session: DbSession) -> Dict[str, Any]:
    """Il volto di una personalità, come il client deve disegnarlo.

    Senza autenticazione sul diritto alla voce: il ritratto si vede anche
    senza poter ascoltare, e nasconderlo a chi non ha il piano giusto
    renderebbe il catalogo anonimo proprio a chi sta decidendo se abbonarsi.
    """
    personalita = await session.scalar(
        select(Personality).where(
            Personality.slug == slug, Personality.status == "published",
        )
    )
    if personalita is None:
        raise HTTPException(status_code=404, detail="Personalità non trovata")

    if personalita.avatar_id is None:
        # Non un 404: la personalità esiste e non ha volto, che è lo stato
        # normale. Il client mostra il nome, e distinguere questo caso da
        # «non esiste» è ciò che gli permette di farlo senza sembrare rotto.
        return {"avatar": None}

    avatar = await session.get(Avatar, personalita.avatar_id)
    if avatar is None:
        return {"avatar": None}

    try:
        descrittore = descrivi(avatar.kind, avatar.config)
    except ConfigurazioneAvatarNonValida as exc:
        # Configurato male: si dice, invece di mandare al client qualcosa che
        # non sa disegnare. L'alternativa è una pagina con un riquadro vuoto e
        # un errore nella console del browser, cioè nessuna informazione per
        # chi l'ha configurato.
        logger.error("Avatar «%s» non valido: %s", avatar.slug, exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"L'avatar di questa personalità è configurato male: {exc}",
        ) from exc

    return {
        "avatar": {
            "id": str(avatar.id),
            "slug": avatar.slug,
            "nome": avatar.name,
            **descrittore.to_dict(),
        }
    }


#: Quanto grande può essere una dettatura.
#:
#: Dieci megabyte sono circa due minuti di webm/opus: oltre, è un caricamento
#: travestito da dettatura e va per l'ingestione, dove c'è una coda, un
#: avanzamento e nessuno che aspetti guardando lo schermo.
BYTE_MASSIMI = 10 * 1024 * 1024

#: Sotto questa fiducia conviene far confermare invece di mandare.
#:
#: Whisper produce testo verosimile anche su registrazioni pessime, e una
#: frase inventata con sicurezza è peggio di nessuna frase: il modello
#: risponderebbe a una domanda che nessuno ha fatto.
FIDUCIA_MINIMA = 0.6


@router.post("/voice/listen")
async def ascolta(
    user: CurrentUser,
    session: DbSession,
    stt: Annotated[Any, Depends(get_stt)],
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Trascrive una registrazione. È il ripiego del riconoscimento del browser.

    L'audio non viene conservato: arriva, si trascrive, e sparisce con la
    directory temporanea. Una registrazione della voce di qualcuno è il genere
    di dato che non si tiene «per ora», perché poi resta.
    """
    await _puo_usare_la_voce(session, user)

    if stt is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "La dettatura lato server non è disponibile su questa "
                "installazione. Il riconoscimento del browser, dove c'è, "
                "funziona lo stesso."
            ),
        )

    audio = await file.read(BYTE_MASSIMI + 1)
    if len(audio) > BYTE_MASSIMI:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Registrazione troppo lunga per una dettatura: oltre un paio "
                "di minuti conviene caricare il file come documento."
            ),
        )

    try:
        esito = await stt.ascolta(
            audio, media_type=file.content_type or "", lingua="it",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dettatura non riuscita")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Non sono riuscito a trascrivere: {exc}",
        ) from exc

    return {
        **esito.to_dict(),
        # Vero quando il testo va mostrato e fatto confermare invece che
        # mandato: è la differenza fra «hai detto questo?» e una risposta
        # perfetta a una domanda che nessuno ha fatto.
        "da_confermare": esito.fiducia < FIDUCIA_MINIMA or not esito.testo,
    }
