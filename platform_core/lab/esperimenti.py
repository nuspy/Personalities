"""Esperimenti fra versioni di una personalità.

**L'assegnazione è una funzione, non uno stato.** La variante di un utente si
calcola da `sha256(esperimento:utente)`: stessa persona, stesso esperimento,
stessa variante, sempre — senza una tabella di assegnazioni da tenere
allineata, e senza che un secondo processo possa sceglierne un'altra nello
stesso istante. Cambiare i pesi a esperimento in corso riassegnerebbe una
parte degli utenti, ed è per questo che i pesi non si cambiano: si conclude e
se ne apre un altro.

**Si misura sulle conversazioni che l'esperimento ha aperto**, non su tutte
quelle servite da una versione: la stessa versione può essere stata la
corrente prima dell'esperimento, e mescolare quei turni confronterebbe
periodi diversi invece di varianti.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.knowledge_models import Personality, PersonalityVersion
from ..domain.lab_models import Experiment
from .statistica import VOTI_MINIMI, confronta, tasso

logger = logging.getLogger(__name__)

ETICHETTE = "ABCDEFGH"


class EsperimentoNonValido(ValueError):
    pass


@dataclass(frozen=True)
class Variante:
    version_id: uuid.UUID
    peso: int
    etichetta: str


def varianti_di(esperimento: Experiment) -> List[Variante]:
    return [
        Variante(uuid.UUID(v["version_id"]), int(v["peso"]), str(v["etichetta"]))
        for v in esperimento.variants
    ]


def assegna(esperimento_id: uuid.UUID, user_id: int, varianti: Sequence[Variante]) -> Variante:
    """La variante di questo utente in questo esperimento. Sempre la stessa."""
    impronta = hashlib.sha256(f"{esperimento_id}:{user_id}".encode()).digest()
    # Un punto in [0, 1) uniforme: 8 byte bastano, e sono lontani dal
    # distorcere una ripartizione fatta di pochi pesi interi.
    punto = int.from_bytes(impronta[:8], "big") / 2**64
    totale = sum(v.peso for v in varianti)
    soglia = 0.0
    for variante in varianti:
        soglia += variante.peso / totale
        if punto < soglia:
            return variante
    return varianti[-1]


class GestoreEsperimenti:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def attivo_per(self, personality_id: uuid.UUID) -> Optional[Experiment]:
        return (await self._session.execute(
            select(Experiment).where(
                Experiment.personality_id == personality_id,
                Experiment.status == "attivo",
            )
        )).scalar_one_or_none()

    async def versione_per(
        self, personalita_id: uuid.UUID, user_id: int,
    ) -> Optional[tuple[Experiment, PersonalityVersion]]:
        """Esperimento e versione per chi apre una conversazione, se ce n'è uno."""
        esperimento = await self.attivo_per(personalita_id)
        if esperimento is None:
            return None
        variante = assegna(esperimento.id, user_id, varianti_di(esperimento))
        versione = await self._session.get(PersonalityVersion, variante.version_id)
        if versione is None or versione.personality_id != personalita_id:
            # La versione è sparita o appartiene ad altri: servire la corrente
            # e dirlo, invece di far fallire la conversazione di chi non ne sa
            # nulla.
            logger.error(
                "Esperimento %s: la variante %s punta a una versione non valida",
                esperimento.id, variante.etichetta,
            )
            return None
        return esperimento, versione

    async def elenco(self, *, personality_id: Optional[uuid.UUID] = None) -> Sequence[Experiment]:
        query = select(Experiment).order_by(Experiment.started_at.desc())
        if personality_id is not None:
            query = query.where(Experiment.personality_id == personality_id)
        return (await self._session.execute(query)).scalars().all()

    async def crea(
        self,
        personalita: Personality,
        *,
        nome: str,
        ipotesi: Optional[str],
        varianti: Sequence[Dict[str, Any]],
        autore_id: Optional[int],
    ) -> Experiment:
        if len(varianti) < 2:
            raise EsperimentoNonValido("servono almeno due varianti da confrontare")
        if len(varianti) > len(ETICHETTE):
            raise EsperimentoNonValido(f"al massimo {len(ETICHETTE)} varianti")

        ids = [uuid.UUID(str(v["version_id"])) for v in varianti]
        if len(set(ids)) != len(ids):
            raise EsperimentoNonValido("la stessa versione compare due volte")
        pesi = [int(v.get("peso", 1)) for v in varianti]
        if any(p <= 0 for p in pesi):
            raise EsperimentoNonValido("ogni variante deve avere un peso positivo")

        versioni = {
            v.id: v for v in (await self._session.execute(
                select(PersonalityVersion).where(PersonalityVersion.id.in_(ids))
            )).scalars()
        }
        for version_id in ids:
            versione = versioni.get(version_id)
            if versione is None or versione.personality_id != personalita.id:
                raise EsperimentoNonValido(
                    f"la versione {version_id} non appartiene a {personalita.slug}"
                )

        if await self.attivo_per(personalita.id) is not None:
            raise EsperimentoNonValido(
                f"c'è già un esperimento attivo su {personalita.slug}: concludilo "
                f"prima di aprirne un altro"
            )

        esperimento = Experiment(
            personality_id=personalita.id,
            name=nome,
            hypothesis=ipotesi,
            status="attivo",
            variants=[
                {"version_id": str(i), "peso": p, "etichetta": ETICHETTE[n],
                 "versione": versioni[i].version}
                for n, (i, p) in enumerate(zip(ids, pesi))
            ],
            started_at=utcnow(),
            created_by=autore_id,
        )
        self._session.add(esperimento)
        await self._session.flush()
        return esperimento

    async def concludi(
        self,
        esperimento: Experiment,
        *,
        vincitore: Optional[uuid.UUID],
        conclusione: Optional[str],
    ) -> Experiment:
        if esperimento.status != "attivo":
            raise EsperimentoNonValido("l'esperimento è già concluso")
        if vincitore is not None and vincitore not in {v.version_id for v in varianti_di(esperimento)}:
            raise EsperimentoNonValido("il vincitore non è una delle varianti")
        esperimento.status = "concluso"
        esperimento.ended_at = utcnow()
        esperimento.winner_version_id = vincitore
        esperimento.conclusion = conclusione
        await self._session.flush()
        return esperimento

    async def risultati(self, esperimento: Experiment) -> Dict[str, Any]:
        """Per variante: quanto è stata usata, quanto è piaciuta, com'è andata."""
        righe = {
            r.version_id: r for r in (await self._session.execute(text("""
                select c.personality_version_id as version_id,
                       count(distinct c.id) as conversazioni,
                       count(distinct c.owner_id) as utenti,
                       count(m.id) filter (where m.role = 'assistant') as risposte,
                       count(f.message_id) filter (where f.vote = 1) as su,
                       count(f.message_id) filter (where f.vote = -1) as giu
                  from conversations c
                  left join messages m on m.conversation_id = c.id
                  left join answer_feedback f on f.message_id = m.id
                 where c.experiment_id = :e
                 group by c.personality_version_id
            """), {"e": esperimento.id})).all()
        }
        tracce = {
            r.version_id: r for r in (await self._session.execute(text("""
                select c.personality_version_id as version_id,
                       percentile_cont(0.5) within group (
                           order by (t.latency_ms->>'primo_token')::float
                       ) as primo_token_p50,
                       count(*) filter (
                           where jsonb_array_length(coalesce(t.grounding->'riferimenti_inventati', '[]'::jsonb)) > 0
                       ) as con_inventati,
                       count(*) as tracce
                  from conversations c
                  join messages m on m.conversation_id = c.id
                  join answer_traces t on t.message_id = m.id
                 where c.experiment_id = :e
                 group by c.personality_version_id
            """), {"e": esperimento.id})).all()
        }

        varianti = varianti_di(esperimento)
        riferimento = righe.get(varianti[0].version_id)
        uscita = []
        for i, variante in enumerate(varianti):
            r = righe.get(variante.version_id)
            t = tracce.get(variante.version_id)
            su, giu = (int(r.su), int(r.giu)) if r else (0, 0)
            voce: Dict[str, Any] = {
                "etichetta": variante.etichetta,
                "version_id": str(variante.version_id),
                "peso": variante.peso,
                "conversazioni": int(r.conversazioni) if r else 0,
                "utenti": int(r.utenti) if r else 0,
                "risposte": int(r.risposte) if r else 0,
                "su": su,
                "giu": giu,
                "approvazione": tasso(su, su + giu),
                "primo_token_ms_p50": round(t.primo_token_p50) if t and t.primo_token_p50 is not None else None,
                "citazioni_inventate": tasso(int(t.con_inventati), int(t.tracce)) if t else None,
            }
            if i > 0:
                su_rif = int(riferimento.su) if riferimento else 0
                giu_rif = int(riferimento.giu) if riferimento else 0
                voce["confronto"] = confronta(su_rif, su_rif + giu_rif, su, su + giu).to_dict()
            uscita.append(voce)

        return {"varianti": uscita, "voti_minimi": VOTI_MINIMI}


def esperimento_json(e: Experiment) -> Dict[str, Any]:
    return {
        "id": str(e.id),
        "personality_id": str(e.personality_id),
        "nome": e.name,
        "ipotesi": e.hypothesis,
        "stato": e.status,
        "varianti": e.variants,
        "iniziato_il": e.started_at.isoformat(),
        "concluso_il": e.ended_at.isoformat() if e.ended_at else None,
        "vincitore": str(e.winner_version_id) if e.winner_version_id else None,
        "conclusione": e.conclusion,
    }
