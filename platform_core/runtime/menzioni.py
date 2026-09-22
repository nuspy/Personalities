"""Le altre voci chiamate in causa con `@Nome`.

**Solo su menzione esplicita.** Una personalità non consulta gli scritti di
un'altra perché la domanda «sembra» riguardarla: sarebbe un comportamento
implicito, imprevedibile per chi scrive e impossibile da spiegare dopo. Con
`@Seneca` la scelta è di chi scrive, si vede nel testo, e resta nella traccia.

**Solo voci a cui si ha accesso.** Una menzione non è un modo di aggirare il
piano: se la personalità chiamata è di una categoria che il piano non
comprende, o è privata di qualcun altro, non porta niente — e nel secondo caso
nemmeno si dice che esiste.
"""
from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..billing.entitlements import Diritti
from ..domain.knowledge_models import CommercialCategory, Personality
from ..domain.repositories import PersonalityRepository
from .persona_engine import Menzione

#: Quante voci per messaggio. Due bastano a un confronto; oltre, il contesto
#: diventa un'antologia e la voce che risponde sparisce fra le altre.
MENZIONI_MASSIME = 2

#: `@` seguito da lettere, cifre, `_` o `-`, non preceduto da una lettera —
#: così un indirizzo email non diventa una menzione.
_MENZIONE = re.compile(r"(?<![\w.])@([^\W\d][\w\-]{1,60})", re.UNICODE)


def _normalizza(testo: str) -> str:
    semplice = unicodedata.normalize("NFKD", testo).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", semplice.lower()).strip("-")


def trova(testo: str) -> List[str]:
    """I nomi menzionati, normalizzati, nell'ordine in cui compaiono, senza doppi."""
    visti: List[str] = []
    for grezzo in _MENZIONE.findall(testo):
        nome = _normalizza(grezzo)
        if nome and nome not in visti:
            visti.append(nome)
    return visti


@dataclass(frozen=True)
class Esclusa:
    nome: str
    motivo: str


async def risolvi(
    session: AsyncSession,
    testo: str,
    *,
    user_id: int,
    diritti: Optional[Diritti],
    escludi: Optional[uuid.UUID] = None,
) -> tuple[List[Menzione], List[Esclusa]]:
    """Le menzioni che portano passaggi, e quelle rifiutate col motivo.

    `diritti` è `None` dove l'installazione non fa pagare: lì ogni voce
    pubblicata è accessibile.
    """
    nomi = trova(testo)
    if not nomi:
        return [], []

    candidate = (await session.execute(
        select(Personality, CommercialCategory.slug)
        .outerjoin(CommercialCategory, CommercialCategory.id == Personality.commercial_category_id)
        .where(
            Personality.status == "published",
            or_(Personality.visibility == "pubblica", Personality.owner_id == user_id),
        )
    )).all()

    def corrisponde(p: Personality, nome: str) -> bool:
        return nome in (p.slug, _normalizza(p.display_name))

    repo = PersonalityRepository(session)
    menzioni: List[Menzione] = []
    escluse: List[Esclusa] = []
    for nome in nomi:
        trovata = next(((p, c) for p, c in candidate if corrisponde(p, nome)), None)
        if trovata is None:
            # Nessuna voce con quel nome — o una privata di un altro, che per
            # chi scrive non esiste. Una menzione a vuoto è anche solo una
            # chiocciola nel testo, e non merita un avviso.
            continue
        personalita, categoria = trovata
        if escludi is not None and personalita.id == escludi:
            continue
        if diritti is not None and not diritti.puo_usare(categoria):
            escluse.append(Esclusa(
                personalita.display_name,
                "il tuo piano non comprende questa voce",
            ))
            continue
        if len(menzioni) >= MENZIONI_MASSIME:
            escluse.append(Esclusa(
                personalita.display_name,
                f"al massimo {MENZIONI_MASSIME} voci per messaggio",
            ))
            continue
        menzioni.append(Menzione(
            slug=personalita.slug,
            nome=personalita.display_name,
            kb_ids=tuple(await repo.corpora_di(personalita.id)),
        ))
    return menzioni, escluse


def menzioni_json(menzioni: Sequence[Menzione], escluse: Sequence[Esclusa]) -> dict:
    return {
        "incluse": [{"slug": m.slug, "nome": m.nome} for m in menzioni],
        "escluse": [{"nome": e.nome, "motivo": e.motivo} for e in escluse],
    }
