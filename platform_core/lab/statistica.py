"""Quanto fidarsi di una differenza fra due varianti.

Il confronto è fra due proporzioni — le risposte approvate sul totale delle
votate — ed è il test z a due campioni con la varianza raggruppata. Non è
la statistica più raffinata possibile; è quella che si spiega in una riga a
chi deve decidere, e che non inventa certezze dove i voti sono pochi.

**Sotto un minimo di voti non si dichiara niente.** Con dieci voti per parte
una differenza di venti punti esce «significativa» una volta su tante per
puro caso, e chi guarda la console tende a fermare l'esperimento proprio in
quel momento. Il minimo non rende il test esatto: impedisce di leggerlo
quando non dice ancora nulla.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

#: Voti per variante sotto i quali non si dà un verdetto.
VOTI_MINIMI = 30

#: La soglia convenzionale. Scritta qui e non nella console: spostarla dopo
#: aver visto i numeri è il modo classico di trovare ciò che si cercava.
ALFA = 0.05


@dataclass(frozen=True)
class Confronto:
    tasso_riferimento: Optional[float]
    tasso_variante: Optional[float]
    differenza: Optional[float]
    z: Optional[float]
    p: Optional[float]
    verdetto: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasso_riferimento": self.tasso_riferimento,
            "tasso_variante": self.tasso_variante,
            "differenza": self.differenza,
            "z": self.z,
            "p": self.p,
            "verdetto": self.verdetto,
        }


def tasso(favorevoli: int, totale: int) -> Optional[float]:
    return favorevoli / totale if totale else None


def confronta(
    su_riferimento: int, votati_riferimento: int,
    su_variante: int, votati_variante: int,
    *, minimo: int = VOTI_MINIMI, alfa: float = ALFA,
) -> Confronto:
    """Il test z sulle due proporzioni di approvazione."""
    p1 = tasso(su_riferimento, votati_riferimento)
    p2 = tasso(su_variante, votati_variante)

    if min(votati_riferimento, votati_variante) < minimo:
        mancano = max(0, minimo - min(votati_riferimento, votati_variante))
        return Confronto(
            p1, p2, (p2 - p1) if p1 is not None and p2 is not None else None,
            None, None,
            f"voti insufficienti: ne servono almeno {minimo} per variante "
            f"(ne mancano {mancano})",
        )

    raggruppato = (su_riferimento + su_variante) / (votati_riferimento + votati_variante)
    errore = math.sqrt(
        raggruppato * (1 - raggruppato)
        * (1 / votati_riferimento + 1 / votati_variante)
    )
    differenza = p2 - p1
    if errore == 0:
        # Tutti favorevoli o tutti contrari da entrambe le parti: nessuna
        # differenza da spiegare, e il rapporto sarebbe 0/0.
        return Confronto(p1, p2, differenza, 0.0, 1.0, "nessuna differenza")

    z = differenza / errore
    p = math.erfc(abs(z) / math.sqrt(2))
    if p >= alfa:
        verdetto = "differenza non significativa"
    elif differenza > 0:
        verdetto = "la variante è migliore del riferimento"
    else:
        verdetto = "la variante è peggiore del riferimento"
    return Confronto(p1, p2, differenza, z, p, verdetto)
