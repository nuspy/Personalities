"""Personalità e corpora creati dagli utenti, per sé.

Il piano dice quante personalità proprie si possono avere e se si possono
caricare corpora propri (`personalita_proprie`, `corpora_propri`). Ciò che si
crea qui è **privato**: pubblicato — lo si usa — ma visibile solo a chi l'ha
creato, fuori dal catalogo, e ignoto a chiunque altro, per cui uno slug altrui
non esiste.

Il versionamento resta quello delle personalità del catalogo: cambiare il
prompt crea una versione nuova, e le conversazioni già avvenute restano legate
alla versione che le ha prodotte. Un utente che riscrive la sua voce dieci
volte deve poter rileggere cosa rispondeva alla terza.
"""
from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .billing.plans import GestoreAbbonamenti
from .domain.base import utcnow
from .domain.knowledge_models import (
    KnowledgeBase, Personality, PersonalityKnowledgeBase, PersonalityVersion,
)

#: Il prompt di chi crea da sé: abbastanza per una voce ricca, non tanto da
#: diventare il modo di mandare un libro al modello a ogni domanda — per
#: quello ci sono i corpora.
PROMPT_MASSIMO = 8000

#: Le personalità private nascono con il recupero acceso e la verifica
#: deterministica delle citazioni: quella col giudice costa una chiamata per
#: risposta, e la sceglie chi amministra, non chi prova.
CONFIGURAZIONE_INIZIALE = {
    "rag_config": {"max_chunks": 6},
    "guard_config": {"groundcheck": "citations"},
    "memory_config": {"enabled": True, "max_memories": 5},
}


class NonConsentito(Exception):
    """Il piano non lo comprende, o il limite è raggiunto."""

    def __init__(self, messaggio: str, *, serve_piano: bool = True) -> None:
        super().__init__(messaggio)
        self.messaggio = messaggio
        self.serve_piano = serve_piano


@dataclass(frozen=True)
class Permessi:
    corpora: bool
    #: `None` significa senza limite: l'installazione non fa pagare.
    personalita: Optional[int]
    usate: int

    def to_dict(self) -> dict:
        return {
            "corpora_propri": self.corpora,
            "personalita_proprie": self.personalita,
            "personalita_usate": self.usate,
        }


def slug_da(nome: str) -> str:
    """Uno slug leggibile e unico: il nome, più un suffisso casuale.

    Il suffisso perché lo slug è unico su tutta la piattaforma e due utenti
    possono chiamare entrambi la loro voce «Marco Aurelio»; senza, il secondo
    riceverebbe un errore su un nome che per lui è libero.
    """
    base = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:40] or "voce"
    return f"{base}-{uuid.uuid4().hex[:6]}"


class Creazioni:
    def __init__(self, session: AsyncSession, user_id: int) -> None:
        self._session = session
        self._utente = user_id

    async def permessi(self) -> Permessi:
        usate = await self._session.scalar(
            select(func.count()).select_from(Personality).where(
                Personality.owner_id == self._utente,
                Personality.visibility == "privata",
                Personality.status != "archived",
            )
        ) or 0
        gestore = GestoreAbbonamenti(self._session)
        if not await gestore.tariffe_in_vigore():
            # Senza catalogo l'installazione non fa pagare, e non avrebbe
            # senso contingentare ciò che regala — come per le quote.
            return Permessi(corpora=True, personalita=None, usate=int(usate))
        diritti = await gestore.diritti_di(self._utente)
        return Permessi(
            corpora=diritti.corpora_propri,
            personalita=diritti.personalita_proprie,
            usate=int(usate),
        )

    # -- lettura ------------------------------------------------------------

    async def personalita(self) -> Sequence[Personality]:
        return (await self._session.execute(
            select(Personality).where(
                Personality.owner_id == self._utente,
                Personality.visibility == "privata",
                Personality.status != "archived",
            ).order_by(Personality.created_at)
        )).scalars().all()

    async def corpora(self) -> Sequence[KnowledgeBase]:
        return (await self._session.execute(
            select(KnowledgeBase).where(KnowledgeBase.owner_id == self._utente)
            .order_by(KnowledgeBase.created_at)
        )).scalars().all()

    async def sua_personalita(self, personality_id: uuid.UUID) -> Optional[Personality]:
        p = await self._session.get(Personality, personality_id)
        if p is None or p.owner_id != self._utente or p.visibility != "privata" or p.status == "archived":
            return None
        return p

    async def suo_corpus(self, kb_id: uuid.UUID) -> Optional[KnowledgeBase]:
        kb = await self._session.get(KnowledgeBase, kb_id)
        return kb if kb is not None and kb.owner_id == self._utente else None

    async def collegati(self, personalita: Personality) -> list[uuid.UUID]:
        return list((await self._session.execute(
            select(PersonalityKnowledgeBase.kb_id).where(
                PersonalityKnowledgeBase.personality_id == personalita.id,
            )
        )).scalars())

    # -- scrittura ----------------------------------------------------------

    async def crea_personalita(
        self, *, nome: str, descrizione: Optional[str], prompt: str,
    ) -> tuple[Personality, PersonalityVersion]:
        permessi = await self.permessi()
        if permessi.personalita is not None:
            if permessi.personalita <= 0:
                # Senza nomi di piani né numeri: il catalogo è configurabile, e
                # un messaggio che cita «Gold ne consente tre» mente il giorno
                # in cui qualcuno cambia il catalogo.
                raise NonConsentito(
                    "Il tuo piano non comprende personalità proprie: sono "
                    "incluse nei piani superiori."
                )
            if permessi.usate >= permessi.personalita:
                raise NonConsentito(
                    f"Hai già {permessi.usate} personalità proprie, il massimo "
                    f"del tuo piano: archiviane una o passa a un piano superiore."
                )

        personalita = Personality(
            slug=slug_da(nome),
            display_name=nome.strip(),
            description=(descrizione or "").strip() or None,
            status="published",
            visibility="privata",
            owner_id=self._utente,
        )
        self._session.add(personalita)
        await self._session.flush()
        versione = await self._nuova_versione(personalita, prompt, numero=1)
        return personalita, versione

    async def modifica_personalita(
        self,
        personalita: Personality,
        *,
        nome: Optional[str] = None,
        descrizione: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Optional[PersonalityVersion]:
        """Nome e descrizione si cambiano sul posto; il prompt con una versione nuova."""
        if nome is not None and nome.strip():
            personalita.display_name = nome.strip()
        if descrizione is not None:
            personalita.description = descrizione.strip() or None

        nuova = None
        if prompt is not None:
            attuale = await self._session.get(PersonalityVersion, personalita.current_version_id)
            if attuale is None or attuale.system_prompt != prompt:
                ultimo = await self._session.scalar(
                    select(func.max(PersonalityVersion.version)).where(
                        PersonalityVersion.personality_id == personalita.id,
                    )
                ) or 0
                nuova = await self._nuova_versione(personalita, prompt, numero=int(ultimo) + 1, da=attuale)
        await self._session.flush()
        return nuova

    async def _nuova_versione(
        self, personalita: Personality, prompt: str, *, numero: int,
        da: Optional[PersonalityVersion] = None,
    ) -> PersonalityVersion:
        versione = PersonalityVersion(
            personality_id=personalita.id,
            version=numero,
            system_prompt=prompt.strip(),
            behavior_rules=(da.behavior_rules if da else {"regole": []}),
            llm_config=(da.llm_config if da else {}),
            rag_config=(da.rag_config if da else CONFIGURAZIONE_INIZIALE["rag_config"]),
            guard_config=(da.guard_config if da else CONFIGURAZIONE_INIZIALE["guard_config"]),
            memory_config=(da.memory_config if da else CONFIGURAZIONE_INIZIALE["memory_config"]),
            published_at=utcnow(),
            author_id=self._utente,
        )
        self._session.add(versione)
        await self._session.flush()
        personalita.current_version_id = versione.id
        await self._session.flush()
        return versione

    async def archivia(self, personalita: Personality) -> None:
        """Archiviata e non cancellata: le conversazioni vi rimandano ancora."""
        personalita.status = "archived"
        await self._session.flush()

    async def crea_corpus(self, *, nome: str, embed_model: str) -> KnowledgeBase:
        permessi = await self.permessi()
        if not permessi.corpora:
            raise NonConsentito(
                "Il tuo piano non comprende corpora propri: sono inclusi nei "
                "piani superiori."
            )
        kb = KnowledgeBase(
            slug=f"utente-{self._utente}-{slug_da(nome)}"[:80],
            name=nome.strip(),
            kind="corpus",
            visibility="private",
            embed_model=embed_model,
            owner_id=self._utente,
        )
        self._session.add(kb)
        await self._session.flush()
        return kb

    async def collega(self, personalita: Personality, kb_ids: Sequence[uuid.UUID]) -> list[uuid.UUID]:
        """Sostituisce i corpora collegati. Solo corpora propri."""
        propri = {kb.id for kb in await self.corpora()}
        estranei = [k for k in kb_ids if k not in propri]
        if estranei:
            # Non propri o inesistenti: la stessa risposta, per non rivelare
            # quali identificativi di corpora esistano.
            raise NonConsentito("Si possono collegare solo i tuoi corpora.", serve_piano=False)

        attuali = {
            r.kb_id: r for r in (await self._session.execute(
                select(PersonalityKnowledgeBase).where(
                    PersonalityKnowledgeBase.personality_id == personalita.id,
                )
            )).scalars()
        }
        for kb_id, riga in attuali.items():
            if kb_id not in kb_ids:
                await self._session.delete(riga)
        for kb_id in kb_ids:
            if kb_id not in attuali:
                self._session.add(PersonalityKnowledgeBase(
                    personality_id=personalita.id, kb_id=kb_id, role="knowledge",
                ))
        await self._session.flush()
        return list(kb_ids)
