"""Piani, abbonamenti, crediti.

Tre cose distinte, e tenerle separate evita di confonderle quando una
richiesta viene negata: i **diritti** dicono se qualcosa è permesso, i
**limiti** quanto, i **crediti** sono la moneta. «Non hai il piano giusto» e
«non hai crediti» si risolvono in modi diversi.
"""
from .credits import CreditiInsufficienti, Movimento, RegistroCrediti
from .entitlements import (
    Diritti, Verdetto, diritti_da, puo_parlare_con,
)
from .plans import (
    BillingProvider, GestoreAbbonamenti, ProviderDiSviluppo,
)

__all__ = [
    "BillingProvider",
    "CreditiInsufficienti",
    "Diritti",
    "GestoreAbbonamenti",
    "Movimento",
    "ProviderDiSviluppo",
    "RegistroCrediti",
    "Verdetto",
    "diritti_da",
    "puo_parlare_con",
]
