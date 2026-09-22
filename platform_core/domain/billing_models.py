"""Piani, abbonamenti, crediti.

**Il saldo è una somma, non un contatore.** `credit_ledger` è append-only: ogni
accredito e ogni consumo è una riga, e il saldo si ottiene sommandole. Un
contatore aggiornato in place sarebbe più veloce da leggere e perderebbe la
sola cosa che conta quando i conti non tornano — *perché* non tornano. Con il
registro, una sottrazione doppia si vede come due righe; con un contatore,
sparisce dentro un numero.

**I piani portano i loro limiti.** `limits` e `entitlements` sono JSONB e non
colonne perché cambiano più spesso dello schema: aggiungere «quante
personalità può creare» a un piano non deve costare una migrazione, o quel
campo finirà spremuto dentro uno esistente che significava altro.

**Un abbonamento non si cancella: si chiude.** `cancel_at` dice quando smette
di valere, e fino ad allora resta attivo — è ciò che distingue «ha disdetto»
da «non ha mai avuto un abbonamento», e serve sia al servizio sia alla
contabilità.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin

#: Gli stati di un abbonamento.
#:
#: `in_prova` è distinto da `attivo` perché i limiti possono differire e
#: perché la fine di una prova è un momento in cui succede qualcosa —
#: qualcuno paga o smette — mentre la fine di un periodo attivo è un rinnovo.
STATI_ABBONAMENTO = ("in_prova", "attivo", "sospeso", "disdetto", "scaduto")

#: Perché i crediti si muovono. Il motivo è obbligatorio: una riga senza
#: spiegazione rende il registro un elenco di numeri.
MOTIVI = (
    "accredito_piano",      # il rinnovo periodico compreso nell'abbonamento
    "acquisto",             # un pacchetto comprato a parte
    "consumo",              # una risposta generata
    "rimborso",             # una risposta fallita, restituita
    "rettifica",            # correzione manuale, con nota
    "scadenza",             # crediti non usati che decadono
)


class Plan(Base, TimestampMixin):
    """Un piano commerciale."""

    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    #: In centesimi, e interi: i decimali in virgola mobile sul denaro
    #: producono differenze che nessuno sa spiegare a fine mese.
    price_monthly: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    price_yearly: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="EUR", nullable=False)

    #: Quanti crediti il piano accredita a ogni periodo.
    credits_per_period: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
    )

    #: Quanto si può fare: messaggi al giorno, personalità create, dimensione
    #: dei corpora. JSONB perché l'elenco cresce senza toccare lo schema.
    limits: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    #: Cosa si può fare: quali categorie commerciali di personalità, se la
    #: voce è inclusa, se si possono caricare documenti propri.
    entitlements: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    #: Ordine di presentazione e di confronto. Un piano con rango maggiore
    #: comprende le categorie di quelli sotto.
    rank: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    #: Un piano ritirato non si cancella: gli abbonamenti che lo citano
    #: devono restare leggibili, e la contabilità dell'anno scorso pure.
    #: `active` dice solo se si può ancora sottoscrivere.
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("ix_plans_rank", "rank"),
    )


class Subscription(Base, TimestampMixin):
    """L'abbonamento di un utente a un piano."""

    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    plan_id: Mapped[int] = mapped_column(
        # `RESTRICT`: un piano citato da un abbonamento non deve poter
        # sparire, o la contabilità perde il riferimento a cosa è stato
        # venduto.
        ForeignKey("plans.id", ondelete="RESTRICT"), nullable=False,
    )

    status: Mapped[str] = mapped_column(String(20), nullable=False)

    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    #: Quando smetterà di valere. Valorizzato alla disdetta: fino a quel
    #: momento l'abbonamento resta attivo, perché è già stato pagato.
    cancel_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    #: L'identificativo presso il fornitore di pagamento, quando ce n'è uno.
    #: Nullo in sviluppo, dove gli abbonamenti si creano a mano.
    external_id: Mapped[Optional[str]] = mapped_column(String(120))

    plan: Mapped[Plan] = relationship(lazy="selectin")

    __table_args__ = (
        CheckConstraint(
            "status in ('in_prova', 'attivo', 'sospeso', 'disdetto', 'scaduto')",
            name="ck_subscriptions_status",
        ),
        CheckConstraint(
            "period_end > period_start", name="ck_subscriptions_periodo",
        ),
        # La query di ogni richiesta: «l'abbonamento attivo di questo utente».
        Index("ix_subscriptions_utente_stato", "user_id", "status"),
    )

    @property
    def vivo(self) -> bool:
        return self.status in ("in_prova", "attivo")


class CreditEntry(Base):
    """Un movimento di crediti. Append-only.

    Non c'è un metodo per modificarla né per cancellarla, e non è una
    dimenticanza: un registro che si può correggere non è un registro. Uno
    sbaglio si ripara con una riga di `rettifica`, che lascia entrambe visibili.
    """

    __tablename__ = "credit_ledger"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )

    #: Positivo accredita, negativo consuma. Mai zero: una riga che non muove
    #: nulla è rumore in un registro che si legge a mano.
    delta: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(String(30), nullable=False)

    #: Cosa ha causato il movimento. Per un consumo è la conversazione, e
    #: senza di essa «dove sono finiti i miei crediti?» non ha risposta.
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"),
    )
    note: Mapped[Optional[str]] = mapped_column(Text)

    #: Quando questi crediti smettono di valere. Solo per gli accrediti
    #: periodici: quelli comprati non scadono, perché sono stati pagati.
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    __table_args__ = (
        CheckConstraint("delta <> 0", name="ck_credit_delta_non_nullo"),
        CheckConstraint(
            "reason in ('accredito_piano', 'acquisto', 'consumo', 'rimborso', "
            "'rettifica', 'scadenza')",
            name="ck_credit_reason",
        ),
        # Il saldo è una somma su questo indice: senza, ogni risposta
        # costerebbe una scansione del registro di quell'utente.
        Index("ix_credit_ledger_utente", "user_id", "created_at"),
    )


class PaymentCheckout(Base, TimestampMixin):
    """Una sessione di pagamento: l'utente è andato a pagare un piano.

    L'abbonamento **non** nasce qui: nasce quando il fornitore conferma il
    pagamento con un evento firmato. Attivarlo all'apertura del checkout
    significherebbe dare il piano a chiunque apra la pagina e poi la chiuda.
    """

    __tablename__ = "payment_checkouts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("plans.id", ondelete="RESTRICT"), nullable=False,
    )
    annuale: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    #: `aperto` finché l'utente non paga o rinuncia; poi `pagato`,
    #: `annullato` o `scaduto`. Uno stato finale non torna indietro.
    status: Mapped[str] = mapped_column(String(20), default="aperto", nullable=False)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    #: La sessione presso il fornitore, e l'abbonamento che ne nasce.
    external_id: Mapped[Optional[str]] = mapped_column(String(120))
    subscription_external_id: Mapped[Optional[str]] = mapped_column(String(120))
    #: Quanto è stato chiesto, in centesimi, fissato all'apertura: se il
    #: prezzo del piano cambia mentre l'utente paga, vale quello che ha visto.
    importo: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="EUR", nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    plan: Mapped[Plan] = relationship(lazy="selectin")

    __table_args__ = (
        CheckConstraint(
            "status in ('aperto', 'pagato', 'annullato', 'scaduto')",
            name="ck_payment_checkouts_status",
        ),
        Index("ix_payment_checkouts_utente", "user_id", "created_at"),
    )


class PaymentEvent(Base):
    """Un evento ricevuto dal fornitore di pagamento. Append-only.

    La chiave è l'identificativo dell'evento presso il fornitore: i fornitori
    rimandano lo stesso evento finché non ricevono un 200, e senza questa riga
    un pagamento confermato due volte accrediterebbe due periodi.
    """

    __tablename__ = "payment_events"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    tipo: Mapped[str] = mapped_column(String(60), nullable=False)
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
