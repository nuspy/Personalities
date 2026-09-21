"""Che cosa è un passaggio, e a cosa serve.

**La classificazione è funzionale, non bibliografica.** Non interessa se un
testo sia una lettera o un trattato: interessa *a cosa serve quando si
costruisce una personalità*. Un passaggio che mostra come qualcuno costruisce
un ragionamento serve al prompt di sistema; uno che racconta un fatto della
sua vita serve al recupero; uno pieno di note del curatore non serve a nulla e
va tolto di mezzo.

Oggi tutto questo finisce nello stesso mucchio, e il risultato si vede: nel
corpus di Seneca l'apparato dell'edizione del 1802 — «Ut a communibus initium
faciam etc. Ep. lxvii» — compare fra i passaggi citabili, e il modello lo cita
come se fosse prosa sua.

**Due assi, non uno.** *Cosa* dice un passaggio e *chi* lo dice sono domande
indipendenti: un avvenimento raccontato da Seneca vale diversamente dallo
stesso raccontato da Tacito, e diversamente ancora da uno storico del
Novecento. Tenerli su un asse solo costringerebbe a scegliere quale dei due
perdere.

**Le etichette non sono esclusive.** Una lettera contiene insieme lessico,
valori e forma del pensiero: una casella sola butterebbe via due terzi di ciò
che quel passaggio insegna.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class Categoria(str, Enum):
    """A cosa serve un passaggio."""

    #: Come parla: parole scelte, costruzioni ricorrenti, ritmo del periodo.
    LESSICO = "lessico"

    #: Come scrive: immagini, figure, architettura del discorso.
    STILE = "stile"

    #: Come ragiona: da cosa parte, come procede, dove arriva.
    PENSIERO = "pensiero"

    #: Cosa considera bene e male, cosa difende, cosa condanna.
    VALORI = "valori"

    #: Come reagisce, cosa teme, cosa desidera, come sta al mondo.
    PSICOLOGIA = "psicologia"

    #: Un fatto della sua vita: un viaggio, un incarico, una perdita.
    AVVENIMENTO = "avvenimento"

    #: Una descrizione del suo modo di essere, di norma scritta da altri.
    CARATTERE = "carattere"

    #: Il mondo attorno: l'epoca, le persone, i luoghi. Non lui.
    CONTESTO = "contesto"

    #: Note del curatore, numeri di pagina, indici, riferimenti bibliografici.
    #: Non è testo dell'autore e non serve a nessuno a valle.
    APPARATO = "apparato"

    @property
    def insegna_la_voce(self) -> bool:
        """Vero se questo passaggio serve ad addestrare *come* parla.

        È la distinzione che rende il dataset di addestramento diverso da un
        mucchio: si impara la voce da chi la usa, non da chi la descrive.
        """
        return self in (Categoria.LESSICO, Categoria.STILE, Categoria.PENSIERO)

    @property
    def porta_fatti(self) -> bool:
        """Vero se contiene affermazioni verificabili.

        Sono i passaggi per cui il groundcheck ha senso: su un passaggio di
        stile non c'è nulla da fondare.
        """
        return self in (
            Categoria.AVVENIMENTO, Categoria.CONTESTO, Categoria.CARATTERE,
        )

    @property
    def da_scartare(self) -> bool:
        return self is Categoria.APPARATO


class Provenienza(str, Enum):
    """Chi lo dice. Esclusiva, a differenza delle categorie."""

    #: Lo ha scritto la persona stessa. È la fonte che vale di più per la
    #: voce: nessuna descrizione di uno stile insegna quanto lo stile.
    AUTORE = "autore"

    #: Qualcuno che l'ha conosciuta o ne è stato contemporaneo.
    CONTEMPORANEO = "contemporaneo"

    #: Uno storico, un biografo, un commentatore venuto dopo.
    POSTERIORE = "posteriore"

    #: Il curatore dell'edizione: note, introduzioni, apparato critico.
    EDITORIALE = "editoriale"

    IGNOTA = "ignota"

    @property
    def e_primaria(self) -> bool:
        return self is Provenienza.AUTORE

    @property
    def peso(self) -> float:
        """Quanto contano le parole di questa fonte per la voce.

        Non è la sua attendibilità storica — uno storico moderno è spesso più
        affidabile di un contemporaneo — ma quanto avvicina a *come parlava*
        quella persona. Sono due domande diverse, e qui interessa la seconda.
        """
        return {
            Provenienza.AUTORE: 1.0,
            Provenienza.CONTEMPORANEO: 0.6,
            Provenienza.POSTERIORE: 0.4,
            Provenienza.EDITORIALE: 0.1,
            Provenienza.IGNOTA: 0.3,
        }[self]


@dataclass
class Etichettatura:
    """Il verdetto della digestione su un passaggio."""

    #: Categoria → quanto quel passaggio ne è portatore, fra 0 e 1.
    categorie: Dict[Categoria, float] = field(default_factory=dict)
    provenienza: Provenienza = Provenienza.IGNOTA

    #: Quanto vale la pena tenerlo. Bassa non significa sbagliato: significa
    #: che non insegna nulla e occuperebbe un posto fra i risultati.
    qualita: float = 0.5

    #: Perché è stato scartato, quando lo è. Un passaggio tolto senza
    #: spiegazione è indistinguibile da uno perso.
    motivo_scarto: str = ""

    #: Una riga su cosa contiene, per chi ispeziona il corpus dalla console.
    sintesi: str = ""

    #: L'apparato da togliere dal passaggio, quando ce n'è.
    #:
    #: L'apparato non è quasi mai un passaggio a sé: è un **prefisso dentro un
    #: passaggio buono** — «*Ut a communibus initium faciam etc. Ep. lxvii .* O
    #: tre volte beati Quelli…». Scartare l'intero passaggio butterebbe via la
    #: prosa dell'autore insieme al riferimento bibliografico.
    #:
    #: Si chiede **cosa togliere** e non il testo ripulito, per due ragioni:
    #: far riscrivere il passaggio raddoppia la risposta — e con quattro
    #: passaggi lunghi il budget si esaurisce a metà JSON — e soprattutto
    #: lascia al modello la possibilità di «migliorare» la prosa, perdendo
    #: proprio il lessico antico che si voleva conservare.
    da_togliere: str = ""

    @property
    def scartato(self) -> bool:
        return bool(self.motivo_scarto)

    @property
    def principale(self) -> Optional[Categoria]:
        """La categoria dominante, se ce n'è una."""
        if not self.categorie:
            return None
        return max(self.categorie.items(), key=lambda kv: kv[1])[0]

    @property
    def insegna_la_voce(self) -> bool:
        """Vero se serve al dataset di addestramento.

        Basta che una categoria di voce sia presente con un punteggio
        consistente: un passaggio che è per metà racconto e per metà lessico
        insegna comunque come parla.
        """
        return any(
            c.insegna_la_voce and p >= 0.4 for c, p in self.categorie.items()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categorie": {c.value: round(p, 3) for c, p in self.categorie.items()},
            "principale": self.principale.value if self.principale else None,
            "provenienza": self.provenienza.value,
            "qualita": round(self.qualita, 3),
            "motivo_scarto": self.motivo_scarto,
            "sintesi": self.sintesi,
            "da_togliere": self.da_togliere,
        }

    @classmethod
    def from_dict(cls, dati: Optional[Mapping[str, Any]]) -> "Etichettatura":
        if not dati:
            return cls()
        return cls(
            categorie={
                categoria_da(k): float(v)
                for k, v in (dati.get("categorie") or {}).items()
                if categoria_da(k) is not None
            },
            provenienza=provenienza_da(dati.get("provenienza")),
            qualita=float(dati.get("qualita", 0.5)),
            motivo_scarto=str(dati.get("motivo_scarto") or ""),
            sintesi=str(dati.get("sintesi") or ""),
        )


def categoria_da(valore: Any) -> Optional[Categoria]:
    """Interpreta l'etichetta restituita dal classificatore.

    `None` e non un ripiego: un'etichetta che non si riconosce va ignorata,
    non tradotta a forza in una categoria vicina. Forzarla significherebbe
    attribuire a un passaggio una natura che nessuno gli ha riconosciuto.
    """
    if not valore:
        return None
    normalizzato = str(valore).strip().lower()
    for categoria in Categoria:
        if categoria.value == normalizzato:
            return categoria

    sinonimi = {
        "vocabolario": Categoria.LESSICO, "vocabulary": Categoria.LESSICO,
        "linguaggio": Categoria.LESSICO,
        "style": Categoria.STILE, "retorica": Categoria.STILE,
        "ragionamento": Categoria.PENSIERO, "thought": Categoria.PENSIERO,
        "logica": Categoria.PENSIERO,
        "values": Categoria.VALORI, "morale": Categoria.VALORI,
        "etica": Categoria.VALORI,
        "psychology": Categoria.PSICOLOGIA, "emozioni": Categoria.PSICOLOGIA,
        "event": Categoria.AVVENIMENTO, "biografia": Categoria.AVVENIMENTO,
        "fatto": Categoria.AVVENIMENTO,
        "character": Categoria.CARATTERE, "descrizione": Categoria.CARATTERE,
        "context": Categoria.CONTESTO, "storia": Categoria.CONTESTO,
        "editoriale": Categoria.APPARATO, "note": Categoria.APPARATO,
        "apparatus": Categoria.APPARATO, "metadati": Categoria.APPARATO,
    }
    return sinonimi.get(normalizzato)


def provenienza_da(valore: Any) -> Provenienza:
    """Interpreta la provenienza, ricadendo su `ignota`.

    Qui il ripiego esiste perché la provenienza è esclusiva e serve sempre:
    non sapere chi parla è uno stato legittimo, e `ignota` lo dice.
    """
    if not valore:
        return Provenienza.IGNOTA
    normalizzato = str(valore).strip().lower()
    for provenienza in Provenienza:
        if provenienza.value == normalizzato:
            return provenienza

    sinonimi = {
        "author": Provenienza.AUTORE, "primaria": Provenienza.AUTORE,
        "lui": Provenienza.AUTORE, "stesso": Provenienza.AUTORE,
        "contemporary": Provenienza.CONTEMPORANEO,
        "testimone": Provenienza.CONTEMPORANEO,
        "later": Provenienza.POSTERIORE, "storico": Provenienza.POSTERIORE,
        "secondaria": Provenienza.POSTERIORE,
        "editor": Provenienza.EDITORIALE, "curatore": Provenienza.EDITORIALE,
        "traduttore": Provenienza.EDITORIALE,
    }
    return sinonimi.get(normalizzato, Provenienza.IGNOTA)


#: Le categorie che il classificatore può assegnare, per il prompt.
DESCRIZIONI: Dict[Categoria, str] = {
    Categoria.LESSICO: (
        "mostra come parla: parole tipiche, costruzioni ricorrenti, il ritmo "
        "della frase"
    ),
    Categoria.STILE: (
        "mostra come scrive: immagini, figure, come costruisce un discorso"
    ),
    Categoria.PENSIERO: (
        "mostra come ragiona: da cosa parte, come procede, dove arriva"
    ),
    Categoria.VALORI: (
        "dice cosa considera bene o male, cosa difende, cosa condanna"
    ),
    Categoria.PSICOLOGIA: (
        "dice come reagisce, cosa teme, cosa desidera, come sta al mondo"
    ),
    Categoria.AVVENIMENTO: "racconta un fatto della sua vita",
    Categoria.CARATTERE: (
        "descrive il suo modo di essere, di norma con parole di altri"
    ),
    Categoria.CONTESTO: (
        "parla del mondo attorno — l'epoca, i luoghi, altre persone — e non "
        "di lui"
    ),
    Categoria.APPARATO: (
        "non e' testo dell'autore: note del curatore, numeri di pagina, "
        "indici, riferimenti bibliografici, intestazioni"
    ),
}
