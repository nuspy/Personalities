"""Operazioni di amministrazione, con la traccia incorporata.

**Perché l'audit sta qui e non nei router.** La specifica chiede che ogni
operazione amministrativa lasci un record, e la differenza fra «lo chiede» e
«accade» è dove si scrive quella riga. In un router si dimentica: basta un
endpoint aggiunto in fretta, e quell'operazione diventa invisibile senza che
nulla lo segnali — anzi, il registro sembra completo proprio perché non
mostra un buco.

Qui la scrittura del record è dentro il metodo che compie l'azione. Non si
può fare l'una senza l'altra perché sono la stessa chiamata.

**Sul versionamento.** Una versione pubblicata non si modifica: modificarla
renderebbe irriproducibili le conversazioni che la citano, ed è esattamente
ciò da cui il versionamento deve proteggere. `nuova_versione()` ne crea una
nuova e sposta il puntatore; la precedente resta dov'era, con le sue
conversazioni intatte.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .base import utcnow
from .knowledge_models import (
    Chunk, CommercialCategory, Document, KnowledgeBase, Personality,
    PersonalityKnowledgeBase, PersonalityType, PersonalityTypeMap,
    PersonalityVersion,
)
from .models import AuditLog, User
from ..avatar.engine import valida as valida_avatar
from .avatar_models import Avatar
from .repositories import AuditRepository

logger = logging.getLogger(__name__)


class ContestoAmministrativo:
    """Chi sta agendo e da dove. Accompagna ogni operazione.

    Esiste perché un record di audit senza autore e senza indirizzo è poco
    più di un log: la domanda a cui un registro deve rispondere è «chi», e
    dev'essere presente al momento dell'azione, non ricostruita dopo.
    """

    def __init__(
        self,
        attore: User,
        *,
        ip: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        self.attore = attore
        self.ip = ip
        self.correlation_id = correlation_id


class _ConAudit:
    """Base per i repository che devono lasciare traccia."""

    def __init__(self, session: AsyncSession, contesto: ContestoAmministrativo) -> None:
        self._session = session
        self._contesto = contesto
        self._audit = AuditRepository(session)

    async def _registra(
        self,
        azione: str,
        *,
        tipo: str,
        target_id: Any,
        prima: Optional[Dict[str, Any]] = None,
        dopo: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._audit.record(
            action=azione,
            actor_id=self._contesto.attore.id,
            target_type=tipo,
            target_id=str(target_id),
            before=prima,
            after=dopo,
            ip=self._contesto.ip,
            correlation_id=self._contesto.correlation_id,
        )


def _istantanea_personalita(p: Personality) -> Dict[str, Any]:
    """Lo stato che vale la pena poter confrontare dopo.

    Non tutto l'oggetto: i campi che qualcuno cambia e di cui qualcun altro
    chiederà conto. Un'istantanea completa renderebbe il registro illeggibile
    e non aggiungerebbe nulla — `created_at` non cambia mai.
    """
    return {
        "slug": p.slug,
        "display_name": p.display_name,
        "description": p.description,
        "status": p.status,
        "current_version_id": str(p.current_version_id) if p.current_version_id else None,
        "commercial_category_id": p.commercial_category_id,
    }


class AdminPersonalityRepository(_ConAudit):
    """Gestione delle personalità."""

    async def _carica_relazioni(self, personalita: Personality) -> None:
        """Rende leggibili `types` e `commercial_category`.

        `selectin` li carica da se' quando l'oggetto viene da una query, ma non
        su uno appena costruito o le cui associazioni sono state cambiate con
        un `DELETE`/`INSERT` diretto.
        """
        await self._session.refresh(personalita, ["types", "commercial_category"])

    async def elenco(
        self, *, includi_bozze: bool = True, limite: int = 100
    ) -> Sequence[Personality]:
        query = select(Personality).order_by(Personality.display_name).limit(limite)
        if not includi_bozze:
            query = query.where(Personality.status == "published")
        return (await self._session.execute(query)).scalars().all()

    async def per_id(self, personality_id: uuid.UUID) -> Optional[Personality]:
        return await self._session.get(Personality, personality_id)

    async def per_slug(self, slug: str) -> Optional[Personality]:
        return (await self._session.execute(
            select(Personality).where(Personality.slug == slug)
        )).scalar_one_or_none()

    async def crea(
        self,
        *,
        slug: str,
        display_name: str,
        description: Optional[str] = None,
        commercial_category_id: Optional[int] = None,
    ) -> Personality:
        personalita = Personality(
            slug=slug,
            display_name=display_name,
            description=description,
            commercial_category_id=commercial_category_id,
            # Nasce in bozza, sempre: una voce senza versione pubblicata non
            # può rispondere, e comparire nel catalogo prima di poterlo fare
            # significa offrire qualcosa che fallisce al primo tentativo.
            status="draft",
            owner_id=self._contesto.attore.id,
        )
        self._session.add(personalita)
        await self._session.flush()
        # Un oggetto appena creato non ha le relazioni caricate, e in una
        # sessione asincrona il caricamento pigro non e' possibile: chi legge
        # `personalita.types` subito dopo otterrebbe `MissingGreenlet` invece
        # della lista vuota che si aspetta.
        await self._carica_relazioni(personalita)

        await self._registra(
            "personalita.creata", tipo="personality", target_id=personalita.id,
            dopo=_istantanea_personalita(personalita),
        )
        return personalita

    async def aggiorna(
        self, personalita: Personality, **campi: Any
    ) -> Personality:
        prima = _istantanea_personalita(personalita)

        for nome, valore in campi.items():
            if valore is not None and hasattr(personalita, nome):
                setattr(personalita, nome, valore)
        await self._session.flush()

        dopo = _istantanea_personalita(personalita)
        if prima != dopo:
            await self._registra(
                "personalita.modificata", tipo="personality",
                target_id=personalita.id, prima=prima, dopo=dopo,
            )
        return personalita

    async def versioni(self, personalita: Personality) -> Sequence[PersonalityVersion]:
        return (await self._session.execute(
            select(PersonalityVersion)
            .where(PersonalityVersion.personality_id == personalita.id)
            .order_by(PersonalityVersion.version.desc())
        )).scalars().all()

    async def nuova_versione(
        self,
        personalita: Personality,
        *,
        system_prompt: str,
        behavior_rules: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        rag_config: Optional[Dict[str, Any]] = None,
        guard_config: Optional[Dict[str, Any]] = None,
        pubblica: bool = False,
    ) -> PersonalityVersion:
        """Crea una versione. Le precedenti restano intatte."""
        ultima = (await self._session.execute(
            select(func.max(PersonalityVersion.version))
            .where(PersonalityVersion.personality_id == personalita.id)
        )).scalar_one_or_none()

        versione = PersonalityVersion(
            personality_id=personalita.id,
            version=(ultima or 0) + 1,
            system_prompt=system_prompt,
            behavior_rules=behavior_rules,
            llm_config=llm_config,
            rag_config=rag_config,
            guard_config=guard_config,
            author_id=self._contesto.attore.id,
        )
        self._session.add(versione)
        await self._session.flush()

        await self._registra(
            "personalita.versione_creata", tipo="personality_version",
            target_id=versione.id,
            dopo={"personality_id": str(personalita.id), "version": versione.version},
        )

        if pubblica:
            await self.pubblica(personalita, versione)

        return versione

    async def pubblica(
        self, personalita: Personality, versione: PersonalityVersion
    ) -> None:
        """Rende una versione quella servita."""
        if versione.personality_id != personalita.id:
            raise ValueError("la versione non appartiene a questa personalità")

        prima = _istantanea_personalita(personalita)

        versione.published_at = versione.published_at or utcnow()
        personalita.current_version_id = versione.id
        personalita.status = "published"
        await self._session.flush()

        await self._registra(
            "personalita.pubblicata", tipo="personality",
            target_id=personalita.id,
            prima=prima, dopo=_istantanea_personalita(personalita),
        )

    async def archivia(self, personalita: Personality) -> None:
        """Toglie dal catalogo senza cancellare.

        Le conversazioni già avvenute continuano a riferirsi alle sue versioni,
        e cancellarla le renderebbe illeggibili — o le porterebbe via del tutto,
        che è peggio.
        """
        prima = _istantanea_personalita(personalita)
        personalita.status = "archived"
        await self._session.flush()

        await self._registra(
            "personalita.archiviata", tipo="personality",
            target_id=personalita.id,
            prima=prima, dopo=_istantanea_personalita(personalita),
        )

    async def imposta_tipi(
        self, personalita: Personality, type_ids: Sequence[int]
    ) -> None:
        prima = list((await self._session.execute(
            select(PersonalityTypeMap.type_id).where(
                PersonalityTypeMap.personality_id == personalita.id
            )
        )).scalars())

        await self._session.execute(
            delete(PersonalityTypeMap).where(
                PersonalityTypeMap.personality_id == personalita.id
            )
        )
        for type_id in type_ids:
            self._session.add(PersonalityTypeMap(
                personality_id=personalita.id, type_id=type_id,
            ))
        await self._session.flush()

        await self._carica_relazioni(personalita)

        if sorted(prima) != sorted(type_ids):
            await self._registra(
                "personalita.tipi_modificati", tipo="personality",
                target_id=personalita.id,
                prima={"type_ids": prima}, dopo={"type_ids": list(type_ids)},
            )

    async def collega_corpus(
        self,
        personalita: Personality,
        kb_id: uuid.UUID,
        *,
        role: str = "knowledge",
        max_chunks: int = 6,
        enabled: bool = True,
    ) -> None:
        esistente = (await self._session.execute(
            select(PersonalityKnowledgeBase).where(
                PersonalityKnowledgeBase.personality_id == personalita.id,
                PersonalityKnowledgeBase.kb_id == kb_id,
            )
        )).scalar_one_or_none()

        if esistente is None:
            self._session.add(PersonalityKnowledgeBase(
                personality_id=personalita.id, kb_id=kb_id,
                role=role, max_chunks=max_chunks, enabled=enabled,
            ))
            azione = "personalita.corpus_collegato"
        else:
            esistente.role = role
            esistente.max_chunks = max_chunks
            esistente.enabled = enabled
            azione = "personalita.corpus_modificato"

        await self._session.flush()
        await self._registra(
            azione, tipo="personality", target_id=personalita.id,
            dopo={"kb_id": str(kb_id), "role": role, "enabled": enabled},
        )

    async def scollega_corpus(
        self, personalita: Personality, kb_id: uuid.UUID
    ) -> None:
        await self._session.execute(
            delete(PersonalityKnowledgeBase).where(
                PersonalityKnowledgeBase.personality_id == personalita.id,
                PersonalityKnowledgeBase.kb_id == kb_id,
            )
        )
        await self._session.flush()
        await self._registra(
            "personalita.corpus_scollegato", tipo="personality",
            target_id=personalita.id, prima={"kb_id": str(kb_id)},
        )

    async def corpora(
        self, personalita: Personality
    ) -> Sequence[PersonalityKnowledgeBase]:
        return (await self._session.execute(
            select(PersonalityKnowledgeBase).where(
                PersonalityKnowledgeBase.personality_id == personalita.id
            )
        )).scalars().all()


class AdminKnowledgeRepository(_ConAudit):
    """Gestione delle knowledge base."""

    async def elenco(self, *, limite: int = 100) -> Sequence[KnowledgeBase]:
        return (await self._session.execute(
            select(KnowledgeBase).order_by(KnowledgeBase.name).limit(limite)
        )).scalars().all()

    async def per_id(self, kb_id: uuid.UUID) -> Optional[KnowledgeBase]:
        return await self._session.get(KnowledgeBase, kb_id)

    async def per_slug(self, slug: str) -> Optional[KnowledgeBase]:
        return (await self._session.execute(
            select(KnowledgeBase).where(KnowledgeBase.slug == slug)
        )).scalar_one_or_none()

    async def crea(
        self,
        *,
        slug: str,
        name: str,
        embed_model: str,
        description: Optional[str] = None,
        kind: str = "corpus",
        visibility: str = "private",
    ) -> KnowledgeBase:
        kb = KnowledgeBase(
            slug=slug, name=name, description=description,
            kind=kind, visibility=visibility, embed_model=embed_model,
            owner_id=self._contesto.attore.id,
        )
        self._session.add(kb)
        await self._session.flush()

        await self._registra(
            "kb.creata", tipo="knowledge_base", target_id=kb.id,
            dopo={"slug": slug, "embed_model": embed_model, "kind": kind},
        )
        return kb

    async def documenti(
        self, kb: KnowledgeBase, *, limite: int = 200
    ) -> Sequence[Document]:
        return (await self._session.execute(
            select(Document)
            .where(Document.kb_id == kb.id)
            .order_by(Document.title)
            .limit(limite)
        )).scalars().all()

    async def elimina_documento(
        self, kb: KnowledgeBase, documento: Document
    ) -> None:
        """Toglie un documento e i suoi passaggi.

        Questa è cancellazione vera, non archiviazione: un documento raccolto
        per errore — la pagina sbagliata, un testo di un altro autore —
        continuerebbe altrimenti a comparire fra le fonti di ogni risposta.
        """
        prima = {"titolo": documento.title, "uri": documento.uri}
        passaggi = await self._session.scalar(
            select(func.count()).select_from(Chunk).where(
                Chunk.document_id == documento.id
            )
        )

        await self._session.delete(documento)
        await self._session.flush()

        await self._registra(
            "kb.documento_eliminato", tipo="knowledge_base", target_id=kb.id,
            prima={**prima, "passaggi": int(passaggi or 0)},
        )


    async def registra_caricamento(
        self, kb: KnowledgeBase, *, build_id: uuid.UUID, file: List[Dict[str, Any]],
    ) -> None:
        """Chi ha caricato cosa, e in quale lavoro.

        Si registra al caricamento e non all'indicizzazione: è l'atto di chi
        amministra, ed è il momento in cui un documento sbagliato entra nel
        sistema — da lì in poi lo fa il worker, che non è nessuno.
        """
        await self._registra(
            "kb.documenti_caricati", tipo="knowledge_base", target_id=kb.id,
            dopo={"build_id": str(build_id), "file": file},
        )


class TassonomiaRepository(_ConAudit):
    """Tipi e categorie commerciali: il vocabolario del catalogo."""

    async def tipi(self) -> Sequence[PersonalityType]:
        return (await self._session.execute(
            select(PersonalityType).order_by(PersonalityType.name)
        )).scalars().all()

    async def crea_tipo(
        self, *, slug: str, name: str, description: Optional[str] = None
    ) -> PersonalityType:
        tipo = PersonalityType(slug=slug, name=name, description=description)
        self._session.add(tipo)
        await self._session.flush()
        await self._registra(
            "tipo.creato", tipo="personality_type", target_id=tipo.id,
            dopo={"slug": slug, "name": name},
        )
        return tipo

    async def categorie(self) -> Sequence[CommercialCategory]:
        return (await self._session.execute(
            select(CommercialCategory).order_by(CommercialCategory.rank)
        )).scalars().all()

    async def crea_categoria(
        self, *, slug: str, name: str, rank: int
    ) -> CommercialCategory:
        categoria = CommercialCategory(slug=slug, name=name, rank=rank)
        self._session.add(categoria)
        await self._session.flush()
        await self._registra(
            "categoria.creata", tipo="commercial_category",
            target_id=categoria.id, dopo={"slug": slug, "rank": rank},
        )
        return categoria


class RegistroAuditRepository:
    """Lettura del registro. Solo lettura: non c'è un metodo per modificarlo."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def recenti(
        self,
        *,
        limite: int = 100,
        azione: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> Sequence[AuditLog]:
        query = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limite)
        if azione:
            query = query.where(AuditLog.action == azione)
        if target_type:
            query = query.where(AuditLog.target_type == target_type)
        return (await self._session.execute(query)).scalars().all()


class AvatarRepository(_ConAudit):
    """Gli avatar, con la validazione al salvataggio.

    **Validati qui e non al disegno.** Un avatar rotto scoperto mentre
    qualcuno conversa e' un volto che non compare e un errore nella console
    del browser: nessuna informazione per chi l'ha configurato, e nessun modo
    di collegarla alla modifica che l'ha causata. Al salvataggio il messaggio
    arriva a chi ha appena sbagliato, mentre ha ancora in mano il perche'.
    """

    async def elenco(self) -> Sequence["Avatar"]:
        return (await self._session.execute(
            select(Avatar).order_by(Avatar.name)
        )).scalars().all()

    async def per_id(self, avatar_id: uuid.UUID) -> Optional["Avatar"]:
        return await self._session.get(Avatar, avatar_id)

    async def per_slug(self, slug: str) -> Optional["Avatar"]:
        return (await self._session.execute(
            select(Avatar).where(Avatar.slug == slug)
        )).scalar_one_or_none()

    async def crea(
        self,
        *,
        slug: str,
        name: str,
        kind: str,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> "Avatar":
        valida_avatar(kind, config)

        avatar = Avatar(
            slug=slug, name=name, kind=kind,
            config=config, description=description,
            owner_id=self._contesto.attore.id,
        )
        self._session.add(avatar)
        await self._session.flush()

        await self._registra(
            "avatar.create", tipo="avatar", target_id=avatar.id,
            dopo={"slug": slug, "kind": kind},
        )
        return avatar

    async def modifica(
        self, avatar: "Avatar", **campi: Any,
    ) -> "Avatar":
        prima = {"slug": avatar.slug, "kind": avatar.kind, "config": avatar.config}

        # Validato con i valori **dopo** la modifica, non con quelli passati:
        # cambiare solo il tipo lasciando la vecchia configurazione produce un
        # avatar che nessuno dei due motori sa leggere, e il controllo sui
        # soli campi ricevuti non se ne accorgerebbe.
        tipo = campi.get("kind", avatar.kind)
        config = campi.get("config", avatar.config)
        valida_avatar(tipo, config)

        for campo, valore in campi.items():
            if valore is not None:
                setattr(avatar, campo, valore)
        await self._session.flush()

        await self._registra(
            "avatar.update", tipo="avatar", target_id=avatar.id,
            prima=prima,
            dopo={"slug": avatar.slug, "kind": avatar.kind, "config": avatar.config},
        )
        return avatar

    async def elimina(self, avatar: "Avatar") -> int:
        """Cancella un avatar e toglie il volto a chi lo portava.

        Restituisce quante personalita' sono rimaste senza. Il vincolo e'
        `SET NULL`: cancellare un volto non deve cancellare le voci, e il
        numero serve a chi amministra per sapere cosa ha appena cambiato
        altrove senza doverlo scoprire guardando.
        """
        quante = len((await self._session.execute(
            select(Personality.id).where(Personality.avatar_id == avatar.id)
        )).scalars().all())

        await self._registra(
            "avatar.delete", tipo="avatar", target_id=avatar.id,
            prima={"slug": avatar.slug, "kind": avatar.kind},
            dopo={"personalita_senza_volto": quante},
        )
        await self._session.delete(avatar)
        await self._session.flush()
        return quante
