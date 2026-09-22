"""Il lavoro che nessuno chiede.

Due manutenzioni con lo stesso problema: **nessuno se ne accorge quando non
girano**. Un abbonamento pagato smette di accreditare crediti alla fine del
primo periodo e l'utente lo scopre restando senza; le memorie crescono senza
consolidamento e il recupero peggiora un poco per volta, senza un giorno in
cui si possa dire che si è rotto.

Per questo la passata **riporta cosa ha fatto** invece di limitarsi a farlo:
un lavoro periodico silenzioso è indistinguibile da uno che non parte.

**Una passata per volta, e un errore per utente non ferma gli altri.** Un
abbonamento che non si rinnova per un dato incoerente non deve impedire il
rinnovo di tutti quelli dopo di lui in lista — che è il modo in cui un
difetto di una riga diventa un guasto di tutti.

Non è uno scheduler: è *una passata*, da far girare da `cron`, da un
`CronJob` di Kubernetes o dal ciclo di `__main__`. Scriversi uno scheduler
significherebbe mantenerne uno, e la piattaforma non ne ha bisogno di uno
proprio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..billing.plans import GestoreAbbonamenti
from sqlalchemy import select

from ..domain.billing_models import Subscription
from ..domain.models import User
from ..domain.session import get_session_factory
from ..memory.consolidation import consolida_tutti
from ..observability.tracing import traccia

logger = logging.getLogger(__name__)


@dataclass
class EsitoPassata:
    rinnovati: int = 0
    piani_base_aperti: int = 0
    crediti_accreditati: int = 0
    abbonamenti_chiusi: int = 0
    utenti_consolidati: int = 0
    memorie_chiuse: int = 0
    #: Cosa è andato storto, per riga. Raccolti e non sollevati: la passata
    #: deve finire, e un elenco di fallimenti è ciò che dice a chi guarda se
    #: il problema è di una riga o di tutte.
    errori: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rinnovati": self.rinnovati,
            "piani_base_aperti": self.piani_base_aperti,
            "crediti_accreditati": self.crediti_accreditati,
            "abbonamenti_chiusi": self.abbonamenti_chiusi,
            "utenti_consolidati": self.utenti_consolidati,
            "memorie_chiuse": self.memorie_chiuse,
            "errori": self.errori,
        }


async def rinnova_scaduti(*, limite: int = 100) -> EsitoPassata:
    """Apre il periodo successivo agli abbonamenti che l'hanno finito.

    **Si scorre una lista di identificativi, non di oggetti**, e questa è la
    parte che fa la differenza. Tenendo gli oggetti caricati prima del ciclo,
    il `rollback` che ripara una riga fallita li scade tutti: i successivi
    esplodono al primo attributo pigro — il piano, per dirne uno — con un
    `MissingGreenlet` che non somiglia per niente alla causa. Il tentativo di
    isolare l'errore produceva così il guasto che voleva evitare, e nei log
    restava un rinnovo fallito e novantanove errori incomprensibili.

    Una sessione per riga in più: non è ciò che impedisce la catena — il
    recupero per identificativo lo fa già — ma tiene il lavoro di un
    abbonamento fuori dalla transazione di un altro, e quello vale per sé.
    """
    esito = EsitoPassata()
    fabbrica = get_session_factory()

    # Gli identificativi e basta: un oggetto ORM caricato qui e usato in
    # un'altra sessione è staccato, e ogni suo attributo pigro — il piano, per
    # dirne uno — esploderebbe al primo accesso.
    async with fabbrica() as lettura:
        da_fare = [
            a.id for a in await GestoreAbbonamenti(lettura).da_rinnovare(limite=limite)
        ]

    if not da_fare:
        return esito

    logger.info("%d abbonamenti da rinnovare", len(da_fare))

    for abbonamento_id in da_fare:
        async with fabbrica() as sessione:
            gestore = GestoreAbbonamenti(sessione)
            abbonamento = await sessione.get(Subscription, abbonamento_id)
            if abbonamento is None:
                # Cancellato fra la lettura e adesso: non è un errore, è una
                # corsa normale su una passata lunga.
                continue

            try:
                rinnovo = await gestore.rinnova(abbonamento)
                await sessione.commit()
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Rinnovo fallito per l'abbonamento %s", abbonamento_id,
                )
                esito.errori.append(f"rinnovo {abbonamento_id}: {exc}")
                await sessione.rollback()
                continue

            if rinnovo.chiuso:
                esito.abbonamenti_chiusi += 1
                logger.info(
                    "Abbonamento %s chiuso: %s", abbonamento_id, rinnovo.motivo,
                )
            else:
                esito.rinnovati += 1
                esito.crediti_accreditati += rinnovo.crediti_accreditati

    return esito


async def apri_piani_base_mancanti(*, limite: int = 100) -> EsitoPassata:
    """Dà il piano gratuito a chi non ha nessun abbonamento.

    Normalmente lo fa il primo accesso, alla nascita dell'account. Qui si
    ripara chi è rimasto fuori: gli utenti che esistevano **prima** che quel
    passaggio esistesse, e quelli a cui è andato storto. Senza, un'installazione
    aggiornata lascia tutti i suoi utenti a saldo zero — hanno i diritti del
    piano gratuito, che ricadono sul codice, e non i suoi crediti, che
    richiedono righe in tabella.

    Idempotente: chi un abbonamento ce l'ha non viene toccato.
    """
    esito = EsitoPassata()
    fabbrica = get_session_factory()

    async with fabbrica() as lettura:
        # Solo chi non ha nessun abbonamento vivo. `outerjoin` e non due
        # query: con molti utenti, una interrogazione per ciascuno sarebbe il
        # modo tipico di rendere lenta una manutenzione che gira da sola.
        senza = (await lettura.execute(
            select(User.id)
            .outerjoin(
                Subscription,
                (Subscription.user_id == User.id)
                & Subscription.status.in_(("in_prova", "attivo")),
            )
            .where(Subscription.id.is_(None), User.deleted_at.is_(None))
            .limit(limite)
        )).scalars().all()

    for user_id in senza:
        async with fabbrica() as sessione:
            try:
                if await GestoreAbbonamenti(sessione).apri_piano_base(user_id):
                    esito.piani_base_aperti += 1
                    await sessione.commit()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Piano base non aperto per l'utente %s", user_id)
                esito.errori.append(f"piano base {user_id}: {exc}")
                await sessione.rollback()

    if esito.piani_base_aperti:
        logger.info("%d piani gratuiti aperti", esito.piani_base_aperti)
    return esito


async def consolida_memorie(*, limite_utenti: int = 100) -> EsitoPassata:
    """Riordina le memorie di chi ne ha."""
    esito = EsitoPassata()

    async with get_session_factory()() as session:
        esiti = await consolida_tutti(session, limite_utenti=limite_utenti)
        await session.commit()

    esito.utenti_consolidati = len(esiti)
    # Fuse e superate: le due forme di «questa memoria non va più cercata
    # da sola». Le scadute no — quelle se ne vanno da sole col tempo, e
    # contarle qui farebbe sembrare il consolidamento più efficace di quanto
    # sia.
    esito.memorie_chiuse = sum(e.fuse + e.superate for e in esiti.values())
    return esito


async def passata(
    *, limite: int = 100, rinnovi: bool = True, memorie: bool = True,
) -> EsitoPassata:
    """Una manutenzione completa, e cosa ha prodotto.

    I due lavori sono indipendenti: un consolidamento che esplode non deve
    impedire i rinnovi, e viceversa. Vanno in sessioni separate anche per
    questo — una transazione sola li legherebbe l'uno alla sorte dell'altro.
    """
    totale = EsitoPassata()

    with traccia("manutenzione.passata"):
        if rinnovi:
            try:
                parziale = await rinnova_scaduti(limite=limite)
                totale.rinnovati = parziale.rinnovati
                totale.crediti_accreditati = parziale.crediti_accreditati
                totale.abbonamenti_chiusi = parziale.abbonamenti_chiusi
                totale.errori.extend(parziale.errori)
            except Exception as exc:  # noqa: BLE001
                logger.exception("La passata dei rinnovi è fallita")
                totale.errori.append(f"rinnovi: {exc}")

        try:
            parziale = await apri_piani_base_mancanti(limite=limite)
            totale.piani_base_aperti = parziale.piani_base_aperti
            totale.errori.extend(parziale.errori)
        except Exception as exc:  # noqa: BLE001
            logger.exception("L'apertura dei piani base è fallita")
            totale.errori.append(f"piani base: {exc}")

        if memorie:
            try:
                parziale = await consolida_memorie(limite_utenti=limite)
                totale.utenti_consolidati = parziale.utenti_consolidati
                totale.memorie_chiuse = parziale.memorie_chiuse
            except Exception as exc:  # noqa: BLE001
                logger.exception("Il consolidamento delle memorie è fallito")
                totale.errori.append(f"memorie: {exc}")

    logger.info("Manutenzione: %s", totale.to_dict())
    return totale
