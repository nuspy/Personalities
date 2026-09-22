"""Dal testo alle forme della bocca.

**Il server non genera mai fotogrammi.** Manda l'audio e una lista di forme
con i loro tempi; ad animare è il client. La ragione non è di gusto: un video
di volto parlante costa al server la sintesi *più* il rendering e la banda di
un flusso video per ogni ascoltatore, e si può riusare per nient'altro. Una
lista di visemi pesa qualche kilobyte, si anima a sessanta fotogrammi al
secondo sulla macchina di chi guarda, e funziona identica su un'immagine
ferma, un video in ciclo o un modello 3D.

**Quindici visemi e non quaranta fonemi.** La bocca non distingue `p`, `b` e
`m` — si chiudono le labbra e basta — e pretendere una forma per fonema
produrrebbe un'animazione più nervosa, non più precisa. Il gruppo è quello di
Preston Blair, lo stesso che usano i motori di animazione facciale, così che
un modello 3D preso altrove porti già le sue pose.

**Dalle lettere, non dai fonemi.** Un dizionario fonetico o un G2P neurale
darebbero un risultato migliore su casi difficili, e costerebbero un modello
in più da caricare, da versionare e da spiegare quando sbaglia. In italiano
l'ortografia è quasi fonetica: le regole qui sotto coprono la lingua con un
pugno di eccezioni scritte a mano (`gli`, `gn`, `sc`, `ch`, le doppie), e su
una bocca che si muove per centoventi millisecondi la differenza fra una `e`
aperta e una chiusa nessuno la vede.

La parte che si vede davvero è un'altra: che la bocca si muova **quando** esce
il suono. Per questo i tempi arrivano dall'allineamento dell'audio vero e non
da una stima sulla lunghezza del testo.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: Le forme. Nomi di Preston Blair, quelli che i modelli 3D già conoscono.
#:
#: `X` è la bocca a riposo: serve per i silenzi, e senza di essa una pausa
#: lascerebbe la bocca ferma nell'ultima forma pronunciata — l'effetto per cui
#: un'animazione sembra congelata invece che silenziosa.
VISEMI = (
    "A",   # a
    "E",   # e
    "I",   # i
    "O",   # o
    "U",   # u
    "MBP", # labbra chiuse: m, b, p
    "FV",  # labbro sul dente: f, v
    "L",   # lingua al palato: l, gli
    "TH",  # dentali: t, d, z
    "SS",  # sibilanti: s, sc, z sorda
    "CH",  # postalveolari: c dolce, g dolce, sc(i)
    "KG",  # velari: c dura, g dura, q, k
    "NN",  # nasali: n, gn
    "RR",  # r
    "X",   # riposo
)

#: Quanto dura al minimo una forma, in secondi.
#:
#: Sotto i sessanta millisecondi la bocca non fa in tempo a raggiungere la
#: posizione e l'animazione diventa un tremolio. Le forme più brevi di così si
#: fondono con la precedente invece di essere disegnate: meno fedele alla
#: fonetica, molto più simile a una bocca vera.
DURATA_MINIMA = 0.06

#: Eccezioni ortografiche dell'italiano, in ordine di lunghezza decrescente.
#:
#: L'ordine conta: `sci` va provata prima di `sc`, e `sc` prima di `s`, o la
#: regola corta vincerebbe sempre e i digrammi non esisterebbero.
_DIGRAMMI: Tuple[Tuple[str, str], ...] = (
    ("gli", "L"),
    ("sci", "CH"),
    ("sce", "CH"),
    ("gn", "NN"),
    ("sc", "SS"),
    ("ch", "KG"),
    ("gh", "KG"),
    ("ci", "CH"),
    ("ce", "CH"),
    ("gi", "CH"),
    ("ge", "CH"),
    ("qu", "KG"),
)

_LETTERE: Dict[str, str] = {
    "a": "A", "e": "E", "i": "I", "o": "O", "u": "U",
    "m": "MBP", "b": "MBP", "p": "MBP",
    "f": "FV", "v": "FV",
    "l": "L",
    "t": "TH", "d": "TH",
    "s": "SS", "z": "SS",
    "c": "KG", "g": "KG", "k": "KG", "q": "KG",
    "n": "NN",
    "r": "RR",
    "h": "",   # muta in italiano: non produce forma
    "j": "CH", "y": "I", "w": "U", "x": "KG",
}


@dataclass
class Viseme:
    """Una forma della bocca e quando assumerla."""

    forma: str
    inizio: float
    fine: float

    @property
    def durata(self) -> float:
        return max(0.0, self.fine - self.inizio)

    def to_dict(self) -> Dict[str, object]:
        return {
            "forma": self.forma,
            "inizio": round(self.inizio, 3),
            "fine": round(self.fine, 3),
        }


@dataclass
class Parola:
    """Una parola con i suoi tempi, come la restituisce l'allineamento."""

    testo: str
    inizio: float
    fine: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "testo": self.testo,
            "inizio": round(self.inizio, 3),
            "fine": round(self.fine, 3),
        }


def _normalizza(parola: str) -> str:
    """Minuscole senza accenti, lettere e basta.

    Gli accenti si tolgono perché in italiano non cambiano la forma della
    bocca: «perché» e «perche» hanno la stessa `e` finale per chi guarda.
    """
    piatta = unicodedata.normalize("NFD", parola.lower())
    piatta = "".join(c for c in piatta if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", piatta)


def forme_di(parola: str) -> List[str]:
    """Le forme della bocca per una parola, in ordine.

    Le doppie danno una forma sola: le labbra si chiudono una volta per
    «babbo», non due, e raddoppiarla produrrebbe uno scatto.
    """
    testo = _normalizza(parola)
    forme: List[str] = []
    i = 0

    while i < len(testo):
        for sequenza, forma in _DIGRAMMI:
            if testo.startswith(sequenza, i):
                # La vocale di `ci`, `gi`, `sci` fa parte del digramma e non
                # va riconsumata; quella di `ce`, `ge`, `sce` nemmeno.
                forme.append(forma)
                i += len(sequenza)
                break
        else:
            forma = _LETTERE.get(testo[i], "")
            if forma:
                forme.append(forma)
            i += 1

    # Le consecutive uguali si fondono: doppie ortografiche e gruppi che
    # cadono nello stesso visema.
    unite: List[str] = []
    for forma in forme:
        if not unite or unite[-1] != forma:
            unite.append(forma)
    return unite


def visemi_da(
    parole: Sequence[Parola], *, durata_totale: float = 0.0,
) -> List[Viseme]:
    """Distribuisce le forme della bocca sui tempi delle parole.

    Le forme di una parola si spartiscono la sua durata in parti uguali. Non è
    la fonetica: una vocale tenuta dura più di una consonante di passaggio. È
    però l'approssimazione che sbaglia meno *dove si vede*, perché l'errore
    resta dentro la parola e non sposta l'attacco — e l'attacco è la cosa che
    un occhio nota immediatamente quando è fuori posto.

    Fra una parola e l'altra si mette la bocca a riposo: senza, una pausa
    lascerebbe l'ultima forma congelata sul volto.
    """
    visemi: List[Viseme] = []
    cursore = 0.0

    for parola in parole:
        if parola.inizio > cursore + DURATA_MINIMA:
            visemi.append(Viseme("X", cursore, parola.inizio))

        forme = forme_di(parola.testo)
        durata = max(0.0, parola.fine - parola.inizio)
        if not forme or durata <= 0:
            cursore = max(cursore, parola.fine)
            continue

        passo = durata / len(forme)
        for indice, forma in enumerate(forme):
            inizio = parola.inizio + indice * passo
            visemi.append(Viseme(forma, inizio, inizio + passo))

        cursore = parola.fine

    if durata_totale > cursore + DURATA_MINIMA:
        visemi.append(Viseme("X", cursore, durata_totale))

    return _fondi_brevi(visemi)


def _fondi_brevi(visemi: List[Viseme]) -> List[Viseme]:
    """Assorbe le forme troppo brevi nella precedente.

    Una forma di venti millisecondi non viene mai raggiunta dalla bocca: il
    client interpola verso di essa e riparte prima di arrivarci, e il
    risultato è un tremolio. Assorbirla tiene il tempo complessivo esatto —
    la forma precedente si allunga — e toglie il rumore.
    """
    if not visemi:
        return []

    risultato: List[Viseme] = [visemi[0]]
    for viseme in visemi[1:]:
        ultimo = risultato[-1]
        if viseme.durata < DURATA_MINIMA or viseme.forma == ultimo.forma:
            ultimo.fine = viseme.fine
        else:
            risultato.append(viseme)

    # La prima può essere rimasta sotto soglia se era l'unica: si allunga
    # invece di sparire, perché una bocca senza nessuna forma non si muove
    # affatto e sembra un guasto.
    if len(risultato) == 1 and risultato[0].durata < DURATA_MINIMA:
        risultato[0].fine = risultato[0].inizio + DURATA_MINIMA

    return risultato
