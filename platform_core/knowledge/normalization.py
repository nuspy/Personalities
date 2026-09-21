"""Riparare ciò che l'estrazione ha rotto.

**Il capolettera.** Nelle edizioni a stampa la prima lettera di un paragrafo è
grande e decorata, e chi digitalizza la marca come elemento a sé. Estraendo il
testo, quella lettera resta staccata dal resto della parola:

    D opo tanto tempo ho riveduto i tuoi luoghi Pompei…

Non è un dettaglio tipografico. Quel passaggio non si trova più cercando
«dopo» — nell'indice lessicale compare la radice `opo`, che non è una parola
di nessuna lingua — e un adapter addestrato su quel testo impara a scrivere
«D opo».

**Perché non basta una regola sulla forma.** Una maiuscola isolata a inizio
paragrafo è quasi sempre un capolettera, ma non sempre:

    O tre volte beati Quelli, a ch'in faccia ai padri…

Qui «O» è un'interiezione, e unirla darebbe «Otre». Lo stesso per le
congiunzioni — «E che», «E di» — e le preposizioni — «A questo».

**Il corpus è il suo dizionario.** Il discriminante che funziona non guarda la
forma ma la frequenza: se il frammento dopo la lettera *non* compare altrove
come parola, mentre la parola unita sì, era un capolettera. Misurato sul
corpus di Seneca: `opo` compare una volta sola (quella rotta) contro le
diciannove di `dopo`; `che` compare novecentotrentatré volte contro zero di
`eche`. La separazione è netta, e non richiede un dizionario esterno né
sapere in che lingua sia il testo.
"""
from __future__ import annotations

import collections
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Una maiuscola isolata seguita dal resto della parola.
#:
#: Si cerca a inizio testo, dopo un a capo e dopo la fine di una frase —
#: ovunque possa cominciare un paragrafo. Restringersi al solo inizio del
#: passaggio sembrava più prudente, ma non funziona: quando il passaggio
#: comincia con l'apparato del curatore il capolettera è il primo carattere
#: *dopo* quello, e cercarlo solo in testa non ne trova nemmeno uno.
#:
#: Allargare non costa: contro i falsi positivi non protegge la posizione ma
#: il vocabolario, ed è una guardia che vale in ogni punto del testo.
_CAPOLETTERA = re.compile(
    r"(?:^|(?<=\n)|(?<=[.!?;:]\s))([A-ZÀ-Þ])\s+([a-zà-ÿ][\w'’]*)"
)

_PAROLA = re.compile(r"[a-zà-ÿA-ZÀ-Þ]{2,}[\w'’]*")

#: Quante volte la parola unita deve superare il frammento perché la fusione
#: sia sicura.
#:
#: Basta che la superi: i due casi sono così distanti — diciannove contro uno,
#: zero contro novecento — che una soglia più alta escluderebbe solo i
#: capolettera rari, cioè proprio quelli che nessun altro meccanismo
#: recupererebbe.
RAPPORTO_MINIMO = 1.0


@dataclass
class EsitoNormalizzazione:
    testi: List[str] = field(default_factory=list)
    ricongiunti: int = 0
    lasciati: int = 0
    #: Cosa è stato unito e cosa no, per chi vuole controllare senza rileggere
    #: l'intero corpus.
    decisioni: List[Tuple[str, str, bool]] = field(default_factory=list)


def vocabolario(testi: Sequence[str]) -> Dict[str, int]:
    """Quante volte ogni parola compare nel corpus."""
    conteggio: collections.Counter = collections.Counter()
    for testo in testi:
        for parola in _PAROLA.findall(testo.lower()):
            conteggio[parola] += 1
    return conteggio


def ricongiungi_capolettera(testi: Sequence[str]) -> EsitoNormalizzazione:
    """Ricompone le parole spezzate dal capolettera.

    Il vocabolario si costruisce **prima** di modificare qualunque cosa, e sul
    corpus intero: valutare un passaggio per volta non darebbe abbastanza dati
    per distinguere un frammento da una parola, e correggerne uno cambierebbe
    il conteggio per quelli dopo.
    """
    frequenze = vocabolario(testi)
    esito = EsitoNormalizzazione()

    for testo in testi:

        def decidi(corrispondenza: "re.Match") -> str:
            lettera, resto = corrispondenza.group(1), corrispondenza.group(2)
            unita = (lettera + resto).lower()

            # Il frammento compare anche altrove come parola? Allora la
            # lettera isolata era una parola a sé. Si sottrae l'occorrenza
            # corrente, perché il testo rotto contribuisce al proprio
            # conteggio.
            frammento_altrove = max(0, frequenze.get(resto.lower(), 0) - 1)
            unita_altrove = frequenze.get(unita, 0)

            if unita_altrove > frammento_altrove * RAPPORTO_MINIMO:
                esito.ricongiunti += 1
                esito.decisioni.append(
                    (f"{lettera} {resto}", lettera + resto, True)
                )
                return lettera + resto

            esito.lasciati += 1
            esito.decisioni.append((f"{lettera} {resto}", lettera + resto, False))
            return corrispondenza.group(0)

        esito.testi.append(_CAPOLETTERA.sub(decidi, testo))

    if esito.ricongiunti or esito.lasciati:
        logger.info(
            "Capolettera: %d ricongiunti, %d lasciati com'erano",
            esito.ricongiunti, esito.lasciati,
        )
    return esito
