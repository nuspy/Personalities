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
from . import (  # noqa: F401
    avatar_models, base, billing_models, build_models, knowledge_models,
    memory_models, models,
)

from .avatar_models import Avatar
from .base import Base, OwnedMixin, TimestampMixin, utcnow
from .billing_models import CreditEntry, PaymentCheckout, PaymentEvent, Plan, Subscription
from .build_models import Build, BuildEvent, RuntimeBinding
from .knowledge_models import (
    DIMENSIONI_EMBEDDING, AnswerTrace, Chunk, ChunkVector, Document,
    KnowledgeBase, Personality, PersonalityKnowledgeBase, PersonalityVersion,
)
from .memory_models import Memory, MemoryAccess
from .models import AuditLog, Conversation, Message, User

__all__ = [
    "Avatar",
    "DIMENSIONI_EMBEDDING",
    "AnswerTrace",
    "AuditLog",
    "Base",
    "Build",
    "BuildEvent",
    "Chunk",
    "ChunkVector",
    "Conversation",
    "CreditEntry",
    "Document",
    "KnowledgeBase",
    "Memory",
    "MemoryAccess",
    "Message",
    "OwnedMixin",
    "Personality",
    "PaymentCheckout",
    "PaymentEvent",
    "Plan",
    "PersonalityKnowledgeBase",
    "PersonalityVersion",
    "RuntimeBinding",
    "Subscription",
    "TimestampMixin",
    "User",
    "utcnow",
]
