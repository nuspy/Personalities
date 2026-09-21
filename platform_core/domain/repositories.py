"""Accesso ai dati, con l'isolamento fra utenti incorporato.

**Il punto di questo modulo è che il filtro per proprietario non sia opzionale.**
Un endpoint che chiede «la conversazione con questo id» non può ricevere quella
di un altro utente, e il modo per garantirlo non è ricordarsi la `WHERE` in ogni
router: è non avere, in nessun punto del codice, un metodo che restituisca una
conversazione senza sapere di chi la sta cercando.

Per questo i metodi di `ConversationRepository` prendono l'utente come primo
argomento e non hanno alternativa. Un `get(id)` senza proprietario non esiste —
non perché sia sconsigliato, ma perché non è scritto.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Sequence

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.keycloak import Principal
from .base import utcnow
from .models import AuditLog, Conversation, Message, User

logger = logging.getLogger(__name__)


class UserRepository:
    """Gli utenti, sincronizzati da ciò che afferma Keycloak."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_subject(self, subject: str) -> Optional[User]:
        result = await self._session.execute(
            select(User).where(User.keycloak_sub == subject)
        )
        return result.scalar_one_or_none()

    async def ensure(self, principal: Principal) -> User:
        """L'utente corrispondente al token, creandolo alla prima comparsa.

        Non esiste una «registrazione» separata: chi supera la verifica del
        token ha già un account, e questa riga è solo il posto dove appendere
        le sue conversazioni. Aspettare un passaggio esplicito di creazione
        significherebbe solo un modo in più di trovarsi autenticati e senza
        profilo.

        I campi anagrafici si riallineano a ogni accesso perché la verità su
        email e nome sta in Keycloak: se l'utente li cambia lì, una copia
        locale che non si aggiorna diventa una seconda verità, sbagliata.
        """
        utente = await self.get_by_subject(principal.subject)

        if utente is None:
            utente = User(
                keycloak_sub=principal.subject,
                email=principal.email,
                display_name=principal.display_name,
                locale=principal.locale,
            )
            self._session.add(utente)
            await self._session.flush()
            logger.info("Primo accesso di %s: utente creato", principal.subject[:8])
            return utente

        if utente.deleted_at is not None:
            # Riattivare in silenzio nasconderebbe che l'account era stato
            # chiuso. Chi ha il potere di riaprirlo lo fa da console, e resta
            # scritto nel registro.
            raise PermissionError("account disattivato")

        utente.email = principal.email or utente.email
        utente.display_name = principal.display_name or utente.display_name
        utente.locale = principal.locale or utente.locale
        return utente


class ConversationRepository:
    """Le conversazioni, sempre viste dalla prospettiva di un proprietario."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, owner: User, *, title: Optional[str] = None) -> Conversation:
        conversazione = Conversation(owner_id=owner.id, title=title)
        self._session.add(conversazione)
        await self._session.flush()
        return conversazione

    async def get(self, owner: User, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """La conversazione, **se** è di questo utente.

        Restituire `None` e non sollevare un errore di permesso è deliberato:
        chi non possiede una conversazione non deve nemmeno poter distinguere
        «non esiste» da «non è tua». La differenza, ripetuta su molti id, dice
        quali id esistono.
        """
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.owner_id == owner.id,
                Conversation.status != "deleted",
            )
        )
        return result.scalar_one_or_none()

    async def list_recent(
        self, owner: User, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Conversation]:
        result = await self._session.execute(
            select(Conversation)
            .where(
                Conversation.owner_id == owner.id,
                Conversation.status == "active",
            )
            # `nulls_last`: una conversazione appena creata non ha ancora
            # messaggi, e senza questo finirebbe in cima o in fondo secondo il
            # capriccio del database.
            .order_by(desc(Conversation.last_message_at).nulls_last())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def messages(
        self, owner: User, conversation_id: uuid.UUID, *, limit: int = 200
    ) -> List[Message]:
        """I messaggi, raggiunti passando per la verifica di proprietà.

        Il filtro è sulla conversazione, non sul messaggio, perché è lì che
        vive il proprietario — e la `JOIN` rende impossibile scriverne una
        versione che se lo dimentichi.
        """
        result = await self._session.execute(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Conversation.id == conversation_id,
                Conversation.owner_id == owner.id,
            )
            .order_by(Message.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        conversation: Conversation,
        *,
        role: str,
        content: str,
        correlation_id: Optional[str] = None,
        tokens: Optional[dict] = None,
    ) -> Message:
        adesso = utcnow()
        messaggio = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            correlation_id=correlation_id,
            tokens=tokens,
            created_at=adesso,
        )
        self._session.add(messaggio)
        conversation.last_message_at = adesso
        await self._session.flush()
        return messaggio

    async def count(self, owner: User) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(Conversation)
            .where(
                Conversation.owner_id == owner.id,
                Conversation.status == "active",
            )
        )
        return int(result.scalar_one())


class AuditRepository:
    """Il registro. Solo scrittura e lettura, mai modifica."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        action: str,
        actor_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        before: Optional[dict] = None,
        after: Optional[dict] = None,
        ip: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditLog:
        voce = AuditLog(
            actor_id=actor_id,
            action=action,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            before=before,
            after=after,
            ip=ip,
            correlation_id=correlation_id,
            created_at=utcnow(),
        )
        self._session.add(voce)
        return voce

    async def for_target(
        self, target_type: str, target_id: str, *, limit: int = 100
    ) -> Sequence[AuditLog]:
        result = await self._session.execute(
            select(AuditLog)
            .where(
                AuditLog.target_type == target_type,
                AuditLog.target_id == str(target_id),
            )
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
        )
        return result.scalars().all()
