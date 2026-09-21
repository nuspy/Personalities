"""Capire cosa contiene un corpus, passaggio per passaggio.

**Due livelli, e il primo non costa nulla.** Prima di interrogare un modello si
scartano i passaggi che si riconoscono a occhio: numeri di pagina, indici,
righe di sole cifre romane, frammenti troppo brevi per insegnare qualcosa. Su
un'edizione ottocentesca sono una porzione non trascurabile, e pagarli a
quattro secondi l'uno sarebbe spendere per confermare l'ovvio.

Ciò che resta va al classificatore, che risponde a due domande indipendenti —
*cosa* dice questo passaggio, e *chi* lo dice — e assegna un punteggio a
ciascuna categoria invece di sceglierne una.

**Sul costo.** Una personalità si costruisce una volta e si aggiorna di rado:
quattro ore su cinquemila passaggi sono un costo una tantum per una voce che
vivrà anni. Il confronto giusto non è con il risparmio, è con un adapter che
impara a imitare il curatore invece dell'autore.

**Sull'affidabilità.** Un classificatore che nessuno ha misurato è
un'opinione: `valuta_contro()` confronta le sue etichette con un campione
annotato a mano, e senza quel campione non si sa se le cinquemila
classificazioni valgano qualcosa.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm.base import GenerationError, GenerationRequest, Message
from ..observability.tracing import traccia
from .taxonomy import (
    DESCRIZIONI, Categoria, Etichettatura, Provenienza, categoria_da,
    provenienza_da,
)

logger = logging.getLogger(__name__)

#: Sotto questa lunghezza un passaggio non insegna nulla, qualunque cosa
#: contenga: non c'è abbastanza testo perché un lessico emerga o un
#: ragionamento si completi.
CARATTERI_MINIMI = 120

#: Sotto questa qualità un passaggio non entra nel recupero.
SOGLIA_QUALITA = 0.3

#: Quanti passaggi classificare per chiamata.
#:
#: Uno per volta costa una chiamata ciascuno; troppi insieme fanno perdere
#: attenzione al modello e le ultime etichette peggiorano. Quattro è il
#: compromesso, ed è sceso da sei dopo aver visto le risposte troncarsi: con
#: passaggi da millecinquecento caratteri, sei non ci stanno nel budget.
PER_LOTTO = 4

_SOLO_NUMERI = re.compile(r"^[\s\d.,;:()\[\]ivxlcdmIVXLCDM—–-]+$")
_RIGA_INDICE = re.compile(r"^\s*(pag\.?|pagina|cap\.?|capitolo|vol\.?|tomo)\s", re.I)


def _istruzioni() -> str:
    righe = [
        "Analizzi i passaggi di un corpus per costruire la personalita' di chi "
        "lo ha scritto. Per ciascuno rispondi a due domande separate.",
        "",
        "**Cosa contiene** — assegna un punteggio fra 0 e 1 a ogni categoria "
        "che si applica. Le categorie non si escludono: un passaggio puo' "
        "essere insieme lessico, valori e forma del pensiero. Ometti quelle a "
        "zero.",
        "",
    ]
    for categoria, descrizione in DESCRIZIONI.items():
        righe.append(f"- `{categoria.value}` — {descrizione}")

    righe += [
        "",
        "**Chi lo dice** — una sola:",
        "",
        "- `autore` — lo ha scritto la persona stessa;",
        "- `contemporaneo` — qualcuno che l'ha conosciuta;",
        "- `posteriore` — uno storico, un biografo, un commentatore successivo;",
        "- `editoriale` — il curatore dell'edizione: note, introduzioni, "
        "apparato critico;",
        "- `ignota` — non si capisce.",
        "",
        "Assegna anche una `qualita` fra 0 e 1: quanto quel passaggio insegna "
        "davvero qualcosa su quella persona. Un frammento di apparato "
        "editoriale vale vicino a zero; una pagina in cui mostra come ragiona "
        "vale vicino a uno.",
        "",
        "Scrivi una `sintesi` di una riga: cosa contiene, per chi ispezionera' "
        "il corpus.",
        "",
        "**Segnala l'apparato del curatore.** Spesso non e' un passaggio a "
        "se': e' un prefisso dentro un passaggio buono. Va tolto **tutto** "
        "cio' che ha aggiunto chi ha curato l'edizione e che l'autore non ha "
        "scritto:",
        "",
        "- l'incipit nella lingua originale messo come richiamo;",
        "- il numero dell'opera, dell'epistola o del capitolo;",
        "- le abbreviazioni di rimando («etc.», «cfr.», «ibid.»);",
        "- numeri di pagina finiti in mezzo alla prosa.",
        "",
        "Esempio. Il passaggio:",
        "",
        "    Consilio tuo accedo etc. Ep. lxviii . I o concorro nel tuo parere,",
        "    che tu ti debbi ascondere nell'ozio.",
        "",
        "comincia con l'incipit latino e il numero dell'epistola. In "
        "`da_togliere` va **l'intero prefisso**, copiato carattere per "
        "carattere:",
        "",
        "    \"Consilio tuo accedo etc. Ep. lxviii . \"",
        "",
        "Resta «I o concorro nel tuo parere…», che e' la prosa dell'autore. "
        "Non riscrivere il testo e non correggere l'ortografia antica: riporta "
        "solo cio' che va tolto. Se non c'e' nulla da togliere, ometti il "
        "campo.",
        "",
        "Rispondi in JSON, un oggetto per passaggio, nello stesso ordine:",
        "",
        '{"passaggi": [{"n": 1, "categorie": {"lessico": 0.8, "valori": 0.5}, '
        '"provenienza": "autore", "qualita": 0.9, "sintesi": "...", '
        '"da_togliere": "Ep. lxvii ."}]}',
    ]
    return "\n".join(righe)


ISTRUZIONI = _istruzioni()


@dataclass
class EsitoDigestione:
    esaminati: int = 0
    ripuliti: int = 0
    scartati_a_vista: int = 0
    scartati_dal_giudizio: int = 0
    classificati: int = 0
    per_categoria: Dict[str, int] = field(default_factory=dict)
    per_provenienza: Dict[str, int] = field(default_factory=dict)

    @property
    def scartati(self) -> int:
        return self.scartati_a_vista + self.scartati_dal_giudizio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "esaminati": self.esaminati,
            "classificati": self.classificati,
            "ripuliti": self.ripuliti,
            "scartati": {
                "a_vista": self.scartati_a_vista,
                "dal_giudizio": self.scartati_dal_giudizio,
                "totale": self.scartati,
            },
            "per_categoria": self.per_categoria,
            "per_provenienza": self.per_provenienza,
        }


def scarto_a_vista(testo: str) -> str:
    """Il motivo per cui un passaggio non merita una chiamata, o stringa vuota.

    Deterministico e prima di tutto: su un'edizione ottocentesca i numeri di
    pagina e le righe d'indice sono tanti, e pagarli a quattro secondi l'uno
    sarebbe spendere per confermare l'ovvio.
    """
    pulito = testo.strip()

    if len(pulito) < CARATTERI_MINIMI:
        return f"troppo breve ({len(pulito)} caratteri)"
    if _SOLO_NUMERI.match(pulito):
        return "solo numeri e punteggiatura"
    if _RIGA_INDICE.match(pulito):
        return "riga di indice o rimando"

    # Un passaggio fatto per metà di cifre è una tavola o un elenco: non
    # insegna un lessico né un ragionamento.
    cifre = sum(c.isdigit() for c in pulito)
    if cifre > len(pulito) * 0.3:
        return "prevalentemente numerico"

    return ""


class Digestore:
    """Classifica i passaggi di un corpus."""

    def __init__(self, provider, *, per_lotto: int = PER_LOTTO) -> None:
        self._provider = provider
        self._per_lotto = per_lotto

    async def classifica(
        self,
        passaggi: Sequence[str],
        *,
        chi: str = "",
        max_tokens: int = 2500,
    ) -> List[Etichettatura]:
        """Etichetta un lotto di passaggi.

        `chi` è il nome della persona: sapere di chi si tratta cambia il
        giudizio sulla provenienza — senza, «autore» e «posteriore» si
        distinguono solo dal tono, e il tono inganna.
        """
        if not passaggi:
            return []

        contesto = (
            f"Il corpus raccoglie testi che riguardano {chi}.\n\n"
            if chi else ""
        )
        elenco = "\n\n".join(
            f"--- passaggio {i + 1} ---\n{t.strip()}"
            for i, t in enumerate(passaggi)
        )

        try:
            with traccia("digestione.classifica", passaggi=len(passaggi)):
                dati = await self._provider.complete_json(GenerationRequest(
                    messages=[
                        Message(role="system", content=ISTRUZIONI),
                        Message(role="user", content=contesto + elenco),
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                ))
        except (GenerationError, ValueError) as exc:
            # Una classificazione fallita non deve perdere i passaggi: restano
            # senza etichetta e li si riprende al giro successivo. Scartarli
            # sarebbe il modo peggiore di reagire a un guasto temporaneo.
            logger.warning("Classificazione non riuscita: %s", exc)
            return [Etichettatura() for _ in passaggi]

        return _interpreta(dati, len(passaggi))


def _interpreta(dati: Any, attesi: int) -> List[Etichettatura]:
    """Legge il verdetto, tollerando le forme che i modelli producono."""
    if isinstance(dati, list):
        grezzi = dati
    elif isinstance(dati, dict):
        grezzi = dati.get("passaggi") or dati.get("passages") or dati.get("items") or []
    else:
        grezzi = []

    per_indice: Dict[int, Etichettatura] = {}

    for posizione, voce in enumerate(grezzi):
        if not isinstance(voce, dict):
            continue

        # `n` se c'è, altrimenti la posizione: i modelli a volte lo omettono, e
        # perdere tutto per un campo mancante sarebbe sproporzionato.
        try:
            indice = int(voce.get("n", posizione + 1)) - 1
        except (TypeError, ValueError):
            indice = posizione

        categorie: Dict[Categoria, float] = {}
        for chiave, valore in (voce.get("categorie") or voce.get("categories") or {}).items():
            categoria = categoria_da(chiave)
            if categoria is None:
                continue
            try:
                punteggio = max(0.0, min(1.0, float(valore)))
            except (TypeError, ValueError):
                continue
            if punteggio > 0:
                categorie[categoria] = punteggio

        etichettatura = Etichettatura(
            categorie=categorie,
            provenienza=provenienza_da(
                voce.get("provenienza") or voce.get("provenance")
            ),
            qualita=_numero(voce.get("qualita", voce.get("quality")), 0.5),
            sintesi=str(voce.get("sintesi") or voce.get("summary") or "").strip(),
            da_togliere=str(
                voce.get("da_togliere") or voce.get("to_remove") or ""
            ).strip(),
        )

        principale = etichettatura.principale
        if principale is not None and principale.da_scartare:
            etichettatura.motivo_scarto = "apparato editoriale, non testo dell'autore"
        elif etichettatura.qualita < SOGLIA_QUALITA:
            etichettatura.motivo_scarto = (
                f"non insegna abbastanza (qualita' {etichettatura.qualita:.2f})"
            )

        if 0 <= indice < attesi:
            per_indice[indice] = etichettatura

    # I passaggi che il modello ha saltato restano senza etichetta: si
    # riprendono al giro successivo invece di essere scartati per omissione.
    return [per_indice.get(i, Etichettatura()) for i in range(attesi)]


def _numero(valore: Any, predefinito: float) -> float:
    try:
        return max(0.0, min(1.0, float(valore)))
    except (TypeError, ValueError):
        return predefinito


# --- misura dell'affidabilità ---------------------------------------------


@dataclass
class Accordo:
    """Quanto il classificatore concorda con un campione annotato a mano."""

    confronti: int = 0
    provenienza_giusta: int = 0
    categoria_giusta: int = 0
    scarti_concordi: int = 0

    @property
    def accordo_provenienza(self) -> float:
        return self.provenienza_giusta / self.confronti if self.confronti else 0.0

    @property
    def accordo_categoria(self) -> float:
        return self.categoria_giusta / self.confronti if self.confronti else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confronti": self.confronti,
            "accordo_provenienza": round(self.accordo_provenienza, 3),
            "accordo_categoria": round(self.accordo_categoria, 3),
            "scarti_concordi": self.scarti_concordi,
        }


def valuta_contro(
    ottenute: Sequence[Etichettatura],
    attese: Sequence[Etichettatura],
) -> Accordo:
    """Confronta le etichette con un campione annotato a mano.

    Serve a rispondere alla domanda che altrimenti resta aperta: **le
    cinquemila classificazioni valgono qualcosa?** Un classificatore che
    nessuno ha misurato è un'opinione, e una personalità costruita su
    un'opinione non si sa da dove venga.

    Si confronta la categoria **dominante** e non l'intera distribuzione: due
    annotatori umani non concordano sui decimali, e pretenderlo dal modello
    misurerebbe il rumore invece dell'accordo.
    """
    accordo = Accordo()

    for ottenuta, attesa in zip(ottenute, attese):
        accordo.confronti += 1
        if ottenuta.provenienza is attesa.provenienza:
            accordo.provenienza_giusta += 1
        if ottenuta.principale is attesa.principale:
            accordo.categoria_giusta += 1
        if ottenuta.scartato == attesa.scartato:
            accordo.scarti_concordi += 1

    return accordo
