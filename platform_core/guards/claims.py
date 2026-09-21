"""Le affermazioni di una risposta, e come si classificano.

**Questo modulo esiste per un rischio preciso.** Se il controllo di fondatezza
segnala come «non ancorata» ogni frase scritta in carattere — una massima, una
esortazione, un'immagine — l'utente lo spegne, ed è l'esito peggiore possibile:
si perde anche la rilevazione delle invenzioni vere.

La difesa non è una soglia più indulgente: è una distinzione. Solo ciò che
**asserisce un fatto** può essere infondato. Un'esortazione non è né vera né
falsa; una massima non si verifica consultando una fonte; il modo in cui un
personaggio si rivolge a chi legge non è una pretesa sul mondo.

Per questo `Tipo` non è un dettaglio del rapporto: è ciò che decide se
un'affermazione entri nel conteggio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class Tipo(str, Enum):
    """Che genere di cosa è un'affermazione."""

    #: Asserisce qualcosa di verificabile: una data, un nome, un evento, ciò
    #: che qualcuno ha detto. È l'unico tipo che si può fondare o meno.
    FATTO = "fatto"

    #: Esortazione, consiglio, imperativo. Non è vero né falso.
    ESORTAZIONE = "esortazione"

    #: Massima, giudizio di valore, principio. Non si verifica consultando una
    #: fonte: si condivide o no.
    MASSIMA = "massima"

    #: Tono, immagini, modo di rivolgersi. È la voce, ed è precisamente ciò
    #: che il sistema deve produrre: misurarla come pretesa sul mondo
    #: significherebbe punire il lavoro riuscito.
    STILE = "stile"

    #: Il personaggio parla della propria esperienza. Non è verificabile in un
    #: corpus e non pretende di esserlo.
    ESPERIENZA = "esperienza"

    @property
    def verificabile(self) -> bool:
        return self is Tipo.FATTO


@dataclass
class Affermazione:
    """Un pezzo di risposta, con il suo giudizio."""

    testo: str
    tipo: Tipo = Tipo.FATTO
    #: `True` se i passaggi la sostengono. Ha senso solo per i fatti.
    sostenuta: bool = False
    #: Le etichette `[Kn]` che la sostengono, secondo il giudice.
    riferimenti: List[str] = field(default_factory=list)
    nota: str = ""

    @property
    def infondata(self) -> bool:
        """Un fatto che i passaggi non sostengono.

        Solo i fatti possono esserlo: è la riga che protegge lo stile dal
        controllo, e va letta insieme al docstring del modulo.
        """
        return self.tipo.verificabile and not self.sostenuta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "testo": self.testo,
            "tipo": self.tipo.value,
            "sostenuta": self.sostenuta,
            "riferimenti": self.riferimenti,
            "nota": self.nota,
        }


@dataclass
class EsitoGroundcheck:
    """Il verdetto su una risposta."""

    #: `citations` (deterministico) o `nli` (ha interrogato un giudice).
    livello: str = "citations"

    #: Riferimenti citati nella risposta che non erano fra quelli forniti.
    #: Questa è la sola misura che non può sbagliare: è un confronto fra
    #: insiemi, non un giudizio.
    riferimenti_inventati: List[str] = field(default_factory=list)

    affermazioni: List[Affermazione] = field(default_factory=list)

    #: Avvisi dei guardrail diversi dalla fondatezza.
    segnalazioni: List[Dict[str, Any]] = field(default_factory=list)

    #: Se il giudice non ha potuto pronunciarsi. Vuoto quando tutto è andato
    #: bene. Si tiene distinto da «nulla da segnalare»: un controllo che non è
    #: stato eseguito non è un controllo superato.
    non_eseguito: str = ""

    @property
    def fondata(self) -> bool:
        """La risposta è ancorata a ciò che le è stato dato.

        Una risposta senza affermazioni di fatto è fondata: non c'è nulla da
        fondare, ed è il caso normale di una risposta tutta in carattere.
        """
        if self.riferimenti_inventati:
            return False
        return not any(a.infondata for a in self.affermazioni)

    @property
    def fatti(self) -> List[Affermazione]:
        return [a for a in self.affermazioni if a.tipo.verificabile]

    @property
    def infondate(self) -> List[Affermazione]:
        return [a for a in self.affermazioni if a.infondata]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "livello": self.livello,
            "fondata": self.fondata,
            "riferimenti_inventati": self.riferimenti_inventati,
            "affermazioni": [a.to_dict() for a in self.affermazioni],
            "segnalazioni": self.segnalazioni,
            "non_eseguito": self.non_eseguito,
            "conteggi": {
                "fatti": len(self.fatti),
                "infondate": len(self.infondate),
                "totale": len(self.affermazioni),
            },
        }


def tipo_da(valore: Optional[str]) -> Tipo:
    """Interpreta l'etichetta restituita dal giudice.

    Ricade su `STILE` e non su `FATTO` quando l'etichetta è ignota, e la scelta
    è deliberata: un'etichetta che non si riconosce non deve trasformarsi in
    un'accusa di infondatezza. Il costo di sbagliare in questa direzione è
    un'invenzione non rilevata; nell'altra, un sistema che segnala ogni
    risposta riuscita e che qualcuno spegnerà.
    """
    if not valore:
        return Tipo.STILE
    normalizzato = str(valore).strip().lower()
    for tipo in Tipo:
        if tipo.value == normalizzato:
            return tipo
    # Sinonimi che i modelli producono spontaneamente.
    sinonimi = {
        "factual": Tipo.FATTO,
        "fact": Tipo.FATTO,
        "fattuale": Tipo.FATTO,
        "stylistic": Tipo.STILE,
        "stile": Tipo.STILE,
        "voce": Tipo.STILE,
        "directive": Tipo.ESORTAZIONE,
        "consiglio": Tipo.ESORTAZIONE,
        "imperativo": Tipo.ESORTAZIONE,
        "opinion": Tipo.MASSIMA,
        "opinione": Tipo.MASSIMA,
        "giudizio": Tipo.MASSIMA,
        "personale": Tipo.ESPERIENZA,
    }
    return sinonimi.get(normalizzato, Tipo.STILE)
