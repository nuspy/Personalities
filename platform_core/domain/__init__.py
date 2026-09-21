"""Il livello dati.

**Perché qui si importano tutti i moduli dei modelli.** Le tabelle si
riferiscono l'una all'altra per nome — `conversations.personality_id` punta a
`personalities`, `knowledge_bases.owner_id` a `users` — e SQLAlchemy risolve
quei nomi solo fra le tabelle che ha visto. Caricarne uno solo produce
`NoReferencedTableError` al primo uso, con un messaggio che parla di una
tabella «non trovata» mentre il difetto è che non è stata importata.

Python esegue questo file prima di qualunque sottomodulo del package, quindi
anche `from platform_core.domain.models import User` porta con sé tutto il
resto: il vincolo è soddisfatto da qualunque strada si entri.
"""
from . import base, knowledge_models, models  # noqa: F401

from .base import Base, OwnedMixin, TimestampMixin, utcnow
from .knowledge_models import (
    DIMENSIONI_EMBEDDING, AnswerTrace, Chunk, ChunkVector, Document,
    KnowledgeBase, Personality, PersonalityKnowledgeBase, PersonalityVersion,
)
from .models import AuditLog, Conversation, Message, User

__all__ = [
    "DIMENSIONI_EMBEDDING",
    "AnswerTrace",
    "AuditLog",
    "Base",
    "Chunk",
    "ChunkVector",
    "Conversation",
    "Document",
    "KnowledgeBase",
    "Message",
    "OwnedMixin",
    "Personality",
    "PersonalityKnowledgeBase",
    "PersonalityVersion",
    "TimestampMixin",
    "User",
    "utcnow",
]
