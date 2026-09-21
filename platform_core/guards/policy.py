"""Guardrail: prevenzione e rilevazione, nello stesso file.

Un guardrail è un documento Markdown con frontmatter, e ha **due sezioni con
due destini diversi**:

- `## Istruzioni` entra nello strato 0 del prompt, dove previene. Costa una
  volta sola — è nel prefisso, quindi nello sconto — e agisce prima che il
  problema esista;
- `## Verifica` non arriva mai al personaggio: è la rubrica con cui il giudice
  controlla la risposta, dopo. Costa una chiamata, e agisce quando la
  prevenzione non è bastata.

**Perché tenerle nello stesso file.** Sono due facce di una regola sola, e
separarle è il modo sicuro di farle divergere: si aggiorna l'istruzione e si
dimentica il criterio, così il personaggio cambia comportamento e il giudice
continua a misurare quello vecchio. Qui la modifica è una sola.

**Perché Markdown e non codice.** Un guardrail lo scrive chi conosce il
dominio, non chi conosce Python — ed è materiale che cambia spesso, con la
stessa frequenza di un testo e non di un programma.
"""
from __future__ import annotations

import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import yaml

from ..paths import DATA_DIR

logger = logging.getLogger(__name__)

#: Dove vivono i guardrail.
CARTELLA_GUARDRAIL = DATA_DIR / "guards"

#: Separatore del frontmatter.
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

#: Titoli di sezione riconosciuti, in minuscolo.
_SEZIONE_ISTRUZIONI = {"istruzioni", "istruzioni al personaggio", "prevenzione"}
_SEZIONE_VERIFICA = {"verifica", "criteri di verifica", "rilevazione", "rubrica"}


@dataclass(frozen=True)
class Guardrail:
    """Una regola, nelle sue due forme."""

    slug: str
    nome: str
    descrizione: str = ""

    #: Va nello strato 0 del prompt. Vuoto significa «solo rilevazione»: un
    #: guardrail può esistere per misurare senza influenzare, ed è il modo di
    #: scoprire quanto spesso un problema si presenti prima di decidere come
    #: affrontarlo.
    istruzioni: str = ""

    #: La rubrica del giudice. Vuoto significa «solo prevenzione».
    verifica: str = ""

    #: `blocca` ferma la risposta, `segnala` la lascia passare marcandola,
    #: `osserva` non fa nulla di visibile e registra soltanto.
    severita: str = "segnala"

    #: A quali personalità si applica. Vuoto: a tutte.
    applica_a: Sequence[str] = field(default_factory=tuple)

    @property
    def previene(self) -> bool:
        return bool(self.istruzioni.strip())

    @property
    def rileva(self) -> bool:
        return bool(self.verifica.strip())

    def vale_per(self, slug_personalita: Optional[str]) -> bool:
        if not self.applica_a:
            return True
        return slug_personalita in self.applica_a


class GuardrailMalformato(ValueError):
    """Il file non è un guardrail utilizzabile."""


def interpreta(testo: str, *, origine: str = "<memoria>") -> Guardrail:
    """Da testo Markdown a `Guardrail`."""
    corrispondenza = _FRONTMATTER.match(testo)
    if corrispondenza is None:
        raise GuardrailMalformato(
            f"{origine}: manca il frontmatter fra `---`, che è dove stanno "
            f"identificativo, nome e severità"
        )

    try:
        testata: Dict[str, Any] = yaml.safe_load(corrispondenza.group(1)) or {}
    except yaml.YAMLError as exc:
        raise GuardrailMalformato(f"{origine}: frontmatter illeggibile — {exc}") from exc

    if not isinstance(testata, dict):
        raise GuardrailMalformato(f"{origine}: il frontmatter non è un dizionario")

    slug = str(testata.get("slug") or "").strip()
    if not slug:
        raise GuardrailMalformato(f"{origine}: manca `slug`")

    severita = str(testata.get("severita") or "segnala").strip()
    if severita not in ("blocca", "segnala", "osserva"):
        raise GuardrailMalformato(
            f"{origine}: severità '{severita}' sconosciuta "
            f"(blocca, segnala, osserva)"
        )

    sezioni = _sezioni(testo[corrispondenza.end():])

    istruzioni = _prima_che_corrisponde(sezioni, _SEZIONE_ISTRUZIONI)
    verifica = _prima_che_corrisponde(sezioni, _SEZIONE_VERIFICA)

    if not istruzioni and not verifica:
        raise GuardrailMalformato(
            f"{origine}: né `## Istruzioni` né `## Verifica`: un guardrail che "
            f"non previene e non rileva non fa nulla"
        )

    applica_a = testata.get("applica_a") or ()
    if isinstance(applica_a, str):
        applica_a = (applica_a,)

    return Guardrail(
        slug=slug,
        nome=str(testata.get("nome") or slug),
        descrizione=str(testata.get("descrizione") or ""),
        istruzioni=istruzioni,
        verifica=verifica,
        severita=severita,
        applica_a=tuple(applica_a),
    )


def _sezioni(corpo: str) -> Dict[str, str]:
    """Le sezioni `##` del documento, per titolo in minuscolo."""
    risultato: Dict[str, str] = {}
    titoli = list(re.finditer(r"^##\s+(.+?)\s*$", corpo, re.MULTILINE))

    for i, titolo in enumerate(titoli):
        inizio = titolo.end()
        fine = titoli[i + 1].start() if i + 1 < len(titoli) else len(corpo)
        risultato[titolo.group(1).strip().lower()] = corpo[inizio:fine].strip()

    return risultato


def _prima_che_corrisponde(sezioni: Dict[str, str], nomi: set) -> str:
    for titolo, contenuto in sezioni.items():
        if titolo in nomi:
            return contenuto
    return ""


def carica_da(cartella: Optional[pathlib.Path] = None) -> List[Guardrail]:
    """Legge tutti i guardrail da una cartella.

    Un file malformato non fa cadere gli altri: si registra e si prosegue.
    Fermarsi al primo errore significherebbe che un refuso in un guardrail
    accessorio disattiva anche quelli che contano.
    """
    cartella = cartella or CARTELLA_GUARDRAIL
    if not cartella.exists():
        logger.info("Nessuna cartella dei guardrail in %s", cartella)
        return []

    caricati: List[Guardrail] = []
    for percorso in sorted(cartella.glob("*.md")):
        try:
            caricati.append(
                interpreta(percorso.read_text(encoding="utf-8"), origine=percorso.name)
            )
        except GuardrailMalformato as exc:
            logger.error("Guardrail ignorato — %s", exc)

    logger.info("Guardrail caricati: %d da %s", len(caricati), cartella)
    return caricati


class RegistroGuardrail:
    """I guardrail attivi, filtrati per personalità."""

    def __init__(self, guardrail: Optional[Sequence[Guardrail]] = None) -> None:
        self._tutti = list(guardrail) if guardrail is not None else carica_da()

    def __len__(self) -> int:
        return len(self._tutti)

    def per(self, slug_personalita: Optional[str] = None) -> List[Guardrail]:
        return [g for g in self._tutti if g.vale_per(slug_personalita)]

    def istruzioni_per(self, slug_personalita: Optional[str] = None) -> List[str]:
        """Le istruzioni da mettere nello strato 0.

        Ordinate per slug: lo strato 0 dev'essere byte-identico fra richieste,
        e un ordine che dipende da come i file sono stati letti dal disco non
        lo è — su un altro sistema, o dopo un rinomino, cambierebbe.
        """
        return [
            g.istruzioni.strip()
            for g in sorted(self.per(slug_personalita), key=lambda g: g.slug)
            if g.previene
        ]

    def rubriche_per(self, slug_personalita: Optional[str] = None) -> List[Guardrail]:
        return [g for g in self.per(slug_personalita) if g.rileva]
