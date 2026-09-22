"""Il registro dei crediti.

**Il saldo è una somma.** Non c'è un contatore da aggiornare: ogni movimento è
una riga, e il saldo si ottiene sommandole. Costa una query in più per lettura
e restituisce ciò che un contatore perde — *perché* i conti non tornano. Una
sottrazione doppia si vede come due righe; dentro un contatore, sparisce.

**Il problema difficile è la concorrenza.** Due risposte generate insieme per
lo stesso utente leggono lo stesso saldo, entrambe lo trovano sufficiente,
entrambe sottraggono: il saldo finisce sotto zero e il servizio ha regalato
una risposta. Non è un caso raro — due schede aperte bastano.

La difesa è un lock sulla riga dell'utente, preso **prima** di leggere il
saldo. Serializza i movimenti di quell'utente e solo i suoi: due persone
diverse non si aspettano a vicenda.

Le alternative scartate, e perché:

- **un saldo materializzato con vincolo `>= 0`** sarebbe più veloce, ma
  duplica lo stato: il giorno in cui la somma e il contatore divergono —
  e divergono — non si sa quale dei due creda;
- **isolamento `SERIALIZABLE`** risolverebbe senza lock espliciti, al prezzo
  di far fallire transazioni che non c'entrano nulla, che chi chiama dovrebbe
  saper ripetere;
- **un controllo dentro l'`INSERT`** (`WHERE (SELECT sum...) >= costo`) sembra
  atomico e non lo è: in `READ COMMITTED` due transazioni concorrenti vedono
  entrambe il saldo di prima.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.billing_models import CreditEntry
from ..domain.models import User

logger = logging.getLogger(__name__)


class CreditiInsufficienti(Exception):
    """Il saldo non basta per l'operazione richiesta."""

    def __init__(self, servono: int, disponibili: int):
        super().__init__(
            f"servono {servono} crediti, disponibili {disponibili}"
        )
        self.servono = servono
        self.disponibili = disponibili


@dataclass
class Movimento:
    """Una riga del registro, come la legge chi la guarda."""

    delta: int
    reason: str
    quando: datetime
    note: Optional[str] = None
    conversation_id: Optional[uuid.UUID] = None

    def to_dict(self) -> dict:
        return {
            "delta": self.delta,
            "reason": self.reason,
            "quando": self.quando.isoformat(),
            "note": self.note,
            "conversation_id": (
                str(self.conversation_id) if self.conversation_id else None
            ),
        }


class RegistroCrediti:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def saldo(self, user_id: int) -> int:
        """I crediti disponibili adesso.

        Gli accrediti scaduti non contano: un piano che ne dà cento al mese
        non deve farli accumulare all'infinito, o il piano più basso diventa
        col tempo indistinguibile dal più alto.
        """
        totale = await self._session.scalar(
            select(func.coalesce(func.sum(CreditEntry.delta), 0)).where(
                CreditEntry.user_id == user_id,
                (CreditEntry.expires_at.is_(None))
                | (CreditEntry.expires_at > utcnow()),
            )
        )
        return int(totale or 0)

    async def accredita(
        self,
        user_id: int,
        quanti: int,
        *,
        reason: str = "accredito_piano",
        note: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> CreditEntry:
        """Aggiunge crediti. Non serve lock: accreditare non può fallire."""
        if quanti <= 0:
            raise ValueError("un accredito muove una quantità positiva")

        voce = CreditEntry(
            user_id=user_id,
            delta=quanti,
            reason=reason,
            note=note,
            expires_at=expires_at,
            created_at=utcnow(),
        )
        self._session.add(voce)
        await self._session.flush()
        return voce

    async def consuma(
        self,
        user_id: int,
        quanti: int,
        *,
        conversation_id: Optional[uuid.UUID] = None,
        note: Optional[str] = None,
    ) -> CreditEntry:
        """Sottrae crediti, se ci sono.

        **Qui sta il lock.** Si blocca la riga dell'utente prima di leggere il
        saldo: senza, due richieste concorrenti leggono lo stesso valore,
        lo trovano entrambe sufficiente, e il saldo finisce sotto zero.

        Il lock si libera al commit della transazione del chiamante, che è
        anche quando la riga di consumo diventa visibile: fra la verifica e
        la scrittura non c'è istante in cui un'altra transazione possa
        inserirsi.
        """
        if quanti <= 0:
            raise ValueError("un consumo muove una quantità positiva")

        await self._blocca(user_id)

        disponibili = await self.saldo(user_id)
        if disponibili < quanti:
            raise CreditiInsufficienti(quanti, disponibili)

        voce = CreditEntry(
            user_id=user_id,
            delta=-quanti,
            reason="consumo",
            conversation_id=conversation_id,
            note=note,
            created_at=utcnow(),
        )
        self._session.add(voce)
        await self._session.flush()

        logger.debug(
            "Consumati %d crediti dell'utente %s (restano %d)",
            quanti, user_id, disponibili - quanti,
        )
        return voce

    async def rimborsa(
        self,
        user_id: int,
        quanti: int,
        *,
        conversation_id: Optional[uuid.UUID] = None,
        note: Optional[str] = None,
    ) -> CreditEntry:
        """Restituisce crediti per qualcosa che non è riuscito.

        Riga propria e non cancellazione del consumo: chi legge il registro
        deve vedere che una risposta è stata tentata, è fallita, ed è stata
        restituita. Cancellando, quella storia sparirebbe e il saldo tornerebbe
        giusto senza spiegare nulla.
        """
        voce = CreditEntry(
            user_id=user_id,
            delta=quanti,
            reason="rimborso",
            conversation_id=conversation_id,
            note=note,
            created_at=utcnow(),
        )
        self._session.add(voce)
        await self._session.flush()
        return voce

    async def rettifica(
        self, user_id: int, delta: int, *, note: str,
    ) -> CreditEntry:
        """Correzione manuale, con la sua ragione obbligatoria.

        È il modo di riparare uno sbaglio senza toccare ciò che è già scritto:
        entrambe le righe restano, e la storia resta leggibile.
        """
        if not note.strip():
            raise ValueError("una rettifica senza motivo non è una rettifica")
        if delta == 0:
            raise ValueError("una rettifica che non muove nulla è rumore")

        voce = CreditEntry(
            user_id=user_id,
            delta=delta,
            reason="rettifica",
            note=note,
            created_at=utcnow(),
        )
        self._session.add(voce)
        await self._session.flush()
        return voce

    async def movimenti(
        self, user_id: int, *, limite: int = 100
    ) -> List[Movimento]:
        righe = (await self._session.execute(
            select(CreditEntry)
            .where(CreditEntry.user_id == user_id)
            .order_by(CreditEntry.created_at.desc(), CreditEntry.id.desc())
            .limit(limite)
        )).scalars().all()

        return [
            Movimento(
                delta=r.delta, reason=r.reason, quando=r.created_at,
                note=r.note, conversation_id=r.conversation_id,
            )
            for r in righe
        ]

    async def _blocca(self, user_id: int) -> None:
        """Prende il lock sulla riga dell'utente.

        `FOR UPDATE` e non un lock consultivo: la riga esiste già, il lock
        muore con la transazione anche se il processo cade, e non c'è nessun
        identificativo numerico da inventare e da far coincidere fra moduli
        diversi.
        """
        await self._session.execute(
            select(User.id).where(User.id == user_id).with_for_update()
        )
