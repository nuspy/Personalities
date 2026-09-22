"""Cosa un utente può fare, e quanto.

Due domande distinte, e tenerle separate evita di confonderle quando una
risposta viene negata.

**I diritti** (`entitlements`) dicono *se* qualcosa è permesso: parlare con
una personalità di categoria Gold, caricare documenti propri, usare la voce.
Sono booleani o insiemi, e non si esauriscono.

**I limiti** (`limits`) dicono *quanto*: messaggi al giorno, personalità
create, dimensione dei corpora. Si esauriscono e si rinnovano.

**I crediti sono una terza cosa ancora**, e stanno nel registro: sono la
moneta, non il permesso. Un utente può avere il diritto di parlare con una
voce e non avere crediti per farlo — e i due «no» vanno detti in modo diverso,
perché si risolvono in modi diversi.

**Senza abbonamento si ricade sul piano gratuito**, non sul nulla. Un servizio
che smette di rispondere a chi non paga è una scelta legittima; farlo senza
dire quale piano servirebbe non lo è.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..domain.billing_models import Plan, Subscription

logger = logging.getLogger(__name__)

#: Lo slug del piano che vale per chi non ha abbonamento.
PIANO_PREDEFINITO = "free"

#: I diritti di chi non ha nulla. Deliberatamente non vuoti: un servizio che
#: non risponde affatto a un utente senza abbonamento non si può nemmeno
#: provare, e la prima esperienza di chiunque comincia da qui.
DIRITTI_BASE: Dict[str, Any] = {
    "categorie": ["free"],
    "voce": False,
    "corpora_propri": False,
    "personalita_proprie": 0,
}

LIMITI_BASE: Dict[str, int] = {
    "messaggi_al_giorno": 20,
    "conversazioni": 5,
    "memorie": 50,
}


@dataclass
class Diritti:
    """Ciò che un utente può fare, risolto."""

    piano: str = PIANO_PREDEFINITO
    categorie: List[str] = field(default_factory=lambda: list(DIRITTI_BASE["categorie"]))
    voce: bool = False
    corpora_propri: bool = False
    personalita_proprie: int = 0
    limiti: Dict[str, int] = field(default_factory=lambda: dict(LIMITI_BASE))
    #: Vero quando si stanno applicando i diritti base per assenza di
    #: abbonamento. Serve a spiegare un rifiuto: «il tuo piano non lo
    #: comprende» e «non hai un piano» richiedono azioni diverse.
    predefiniti: bool = True

    def puo_usare(self, categoria: Optional[str]) -> bool:
        """Vero se l'utente può parlare con una personalità di quella categoria.

        Una personalità senza categoria è accessibile a tutti: è lo stato in
        cui nasce, e negarla significherebbe rendere invisibile ogni voce che
        nessuno ha ancora collocato.
        """
        if not categoria:
            return True
        return categoria in self.categorie

    def limite(self, nome: str) -> Optional[int]:
        """Il limite, o `None` se non ce n'è uno.

        `None` e non un numero grandissimo: «illimitato» e «un milione» si
        comportano allo stesso modo finché qualcuno non arriva a un milione.
        """
        valore = self.limiti.get(nome)
        if valore is None or valore < 0:
            return None
        return valore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "piano": self.piano,
            "categorie": self.categorie,
            "voce": self.voce,
            "corpora_propri": self.corpora_propri,
            "personalita_proprie": self.personalita_proprie,
            "limiti": self.limiti,
            "predefiniti": self.predefiniti,
        }


def diritti_da(
    abbonamento: Optional[Subscription], *, piano_base: Optional[Plan] = None,
) -> Diritti:
    """Risolve i diritti di un utente dal suo abbonamento.

    Un abbonamento sospeso o scaduto non dà nulla più del piano base: la
    differenza fra «non ha mai pagato» e «ha smesso di pagare» conta per la
    contabilità, non per cosa può fare oggi.
    """
    if abbonamento is None or not abbonamento.vivo:
        return _dal_piano(piano_base) if piano_base else Diritti()

    return _dal_piano(abbonamento.plan, predefiniti=False)


def _dal_piano(piano: Optional[Plan], *, predefiniti: bool = True) -> Diritti:
    if piano is None:
        return Diritti()

    concessioni = piano.entitlements or {}
    limiti = dict(LIMITI_BASE)
    limiti.update(piano.limits or {})

    return Diritti(
        piano=piano.slug,
        # I diritti del piano **sostituiscono** quelli base invece di
        # aggiungersi: un piano che elenca le proprie categorie dice tutto ciò
        # a cui dà accesso, e unire le due liste renderebbe impossibile
        # togliere qualcosa da un piano.
        categorie=list(concessioni.get("categorie", DIRITTI_BASE["categorie"])),
        voce=bool(concessioni.get("voce", False)),
        corpora_propri=bool(concessioni.get("corpora_propri", False)),
        personalita_proprie=int(concessioni.get("personalita_proprie", 0)),
        limiti=limiti,
        predefiniti=predefiniti,
    )


@dataclass
class Verdetto:
    """L'esito di un controllo, con il motivo quando è negativo."""

    permesso: bool
    motivo: str = ""
    #: Cosa servirebbe per ottenere il permesso: il piano da sottoscrivere,
    #: oppure `crediti` quando il diritto c'è e manca la moneta. Senza,
    #: l'utente sa di non poter fare qualcosa e non sa cosa fare.
    serve: str = ""

    def __bool__(self) -> bool:
        return self.permesso


def puo_parlare_con(
    diritti: Diritti,
    categoria: Optional[str],
    *,
    piani_che_la_comprendono: Sequence[str] = (),
) -> Verdetto:
    """Se l'utente può usare una personalità di quella categoria."""
    if diritti.puo_usare(categoria):
        return Verdetto(True)

    if diritti.predefiniti:
        motivo = (
            f"Questa personalità è riservata agli abbonati; "
            f"il tuo accesso è quello gratuito."
        )
    else:
        motivo = (
            f"Il piano «{diritti.piano}» non comprende le personalità "
            f"della categoria «{categoria}»."
        )

    return Verdetto(
        False,
        motivo=motivo,
        serve=piani_che_la_comprendono[0] if piani_che_la_comprendono else "",
    )
