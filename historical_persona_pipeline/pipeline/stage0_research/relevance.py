"""Filtro di pertinenza sui documenti trovati online.

Una ricerca per "Giulio Cesare" restituisce anche *Augusto*, *Germanico
Giulio Cesare* e *Giulio Cesare Vanini*, filosofo del Seicento. Senza un
filtro finiscono tutti nel corpus e il profilo di stile misura una media fra
persone diverse — un difetto che in una pipeline senza supervisione umana non
verrebbe mai notato.

Il punteggio combina tre segnali, nessuno dei quali basta da solo:

1. **corrispondenza del titolo** con il nome cercato (peso maggiore);
2. **densita' del nome** nel testo: una pagina che parla davvero del
   personaggio lo nomina di continuo;
3. **coerenza temporale**: le date citate nel testo devono cadere vicino
   all'epoca dichiarata. E' il segnale che separa due omonimi distanti secoli.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Soglia sotto la quale un documento viene scartato.
DEFAULT_THRESHOLD = 0.40

# Anni in formato 1234, oppure 44 a.C. / 44 BC.
_YEAR = re.compile(r"\b(\d{1,4})\s*(a\.?\s?c\.?|b\.?\s?c\.?|d\.?\s?c\.?|a\.?\s?d\.?)?\b", re.IGNORECASE)
_BCE_MARKERS = ("a.c", "ac", "b.c", "bc", "a. c")

# Secolo dichiarato nell'epoca: "I secolo a.C.", "16th century", "XVI secolo".
_CENTURY_ARABIC = re.compile(r"\b(\d{1,2})\s*(?:°|º|th|st|nd|rd)?\s*(?:secolo|century|siglo|siècle|jahrhundert)", re.IGNORECASE)
_CENTURY_ROMAN = re.compile(r"\b([IVXL]{1,5})\s*(?:secolo|century)", re.IGNORECASE)

_ROMAN_VALUES = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
                 "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
                 "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
                 "XIX": 19, "XX": 20, "XXI": 21}


@dataclass
class RelevanceVerdict:
    score: float
    keep: bool
    reasons: List[str]


class RelevanceFilter:
    def __init__(
        self,
        author_name: str,
        era: str = "",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.author_name = author_name.strip()
        self.era = era
        self.threshold = threshold
        self.name_tokens = [
            t.lower() for t in re.split(r"\s+", self.author_name) if len(t) > 2
        ]
        self.expected_century = self._parse_century(era)

    # ------------------------------------------------------------- pubblico

    def evaluate(self, title: str, text: str, is_primary_source: bool = False) -> RelevanceVerdict:
        """Fonti primarie e secondarie vanno giudicate con criteri diversi.

        In una fonte primaria il titolo e' quello dell'**opera** (*De bello
        Gallico*), non il nome dell'autore: pretendere che contenga il nome
        scarterebbe proprio i testi che servono. La provenienza gia' lega
        quel testo all'autore, quindi qui basta escludere l'incoerenza
        temporale.

        In una fonte secondaria il titolo e' il soggetto della voce, ed e'
        il segnale che separa il personaggio dai suoi omonimi.
        """
        reasons: List[str] = []
        period_score, period_reason = self._period_score(text)
        if period_reason:
            reasons.append(period_reason)

        # Veto temporale: un testo che parla stabilmente di un'altra epoca
        # riguarda un'altra persona, per quanto il nome coincida. E' il solo
        # segnale che separa Giulio Cesare da Giulio Cesare Vanini.
        if period_score == 0.0 and period_reason:
            return RelevanceVerdict(score=0.0, keep=False, reasons=reasons)

        if is_primary_source:
            reasons.append("fonte primaria")
            # Superato il veto, un testo d'autore si tiene: la pertinenza e'
            # garantita da come lo si e' trovato.
            return RelevanceVerdict(score=0.9, keep=True, reasons=reasons)

        title_score = self._title_score(title)
        density_score = self._density_score(text)

        score = 0.5 * title_score + 0.3 * density_score + 0.2 * period_score

        if title_score == 0.0:
            # Il nome non compare nel titolo: e' materiale di contesto
            # (*I secolo a.C.*, *Impero romano*), che cita il personaggio
            # senza trattarne. Utile a un'enciclopedia, fuorviante per
            # misurare una voce.
            score *= 0.6
            reasons.append("contesto, non soggetto")

        reasons.append(f"titolo {title_score:.2f}")
        reasons.append(f"densita' {density_score:.2f}")

        return RelevanceVerdict(
            score=round(min(score, 1.0), 3),
            keep=score >= self.threshold,
            reasons=reasons,
        )

    # -------------------------------------------------------------- segnali

    def _title_score(self, title: str) -> float:
        """Quota dei token del nome presenti nel titolo, con penalita'.

        Un titolo che contiene tutto il nome *piu' altri nomi propri* e'
        quasi sempre un'altra entita': un parente (*Germanico Giulio
        Cesare*), un omonimo (*Giulio Cesare Vanini*) o un'opera che lo
        ritrae (*Giulio Cesare (Shakespeare)*). Ogni parola in piu' pesa
        molto, perche' il nome cercato da solo e' il caso che interessa.
        """
        if not self.name_tokens:
            return 0.0

        title_lower = title.lower()

        # La disambiguazione fra parentesi di Wikipedia segnala un'entita'
        # diversa dal personaggio: un dramma, un film, un luogo.
        has_qualifier = bool(re.search(r"\(.+\)", title))
        title_core = re.sub(r"\(.*?\)", " ", title_lower)
        title_tokens = [t for t in re.split(r"[\s,]+", title_core) if t]

        matched = sum(1 for token in self.name_tokens if token in title_core)
        coverage = matched / len(self.name_tokens)

        if coverage == 0:
            return 0.0

        extra = [
            t for t in title_tokens
            if t not in self.name_tokens and len(t) > 3 and t.isalpha()
        ]
        penalty = min(1.0, 0.5 * len(extra))
        if has_qualifier:
            penalty += 0.5

        return max(0.0, coverage - penalty)

    def _density_score(self, text: str) -> float:
        """Quante volte il nome compare, ogni mille parole."""
        if not self.name_tokens or not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Il token piu' lungo del nome e' il piu' discriminante (il cognome).
        key_token = max(self.name_tokens, key=len)
        occurrences = len(re.findall(rf"\b{re.escape(key_token)}\w{{0,3}}\b", text, re.IGNORECASE))

        per_thousand = occurrences / len(words) * 1000
        # 5 menzioni ogni mille parole indicano che il testo verte sul soggetto.
        return min(1.0, per_thousand / 5.0)

    def _period_score(self, text: str) -> Tuple[float, Optional[str]]:
        """Coerenza fra le date del testo e il secolo dichiarato."""
        if self.expected_century is None:
            return 0.5, None  # nessuna epoca dichiarata: segnale neutro

        centuries = self._extract_centuries(text[:20_000])
        if not centuries:
            return 0.5, None

        # Secolo prevalente fra quelli citati.
        dominant = max(set(centuries), key=centuries.count)
        distance = abs(dominant - self.expected_century)

        if distance <= 1:
            return 1.0, None
        if distance <= 3:
            return 0.5, f"epoca vicina ma non coincidente (secolo {dominant})"
        return 0.0, f"epoca incoerente: il testo parla del secolo {dominant}"

    # ------------------------------------------------------------- utilities

    @staticmethod
    def _parse_century(era: str) -> Optional[int]:
        """Secolo atteso, con segno negativo se avanti Cristo."""
        if not era:
            return None

        era_lower = era.lower()
        is_bce = any(marker in era_lower.replace(" ", "") for marker in ("a.c", "ac.", "bc", "bce"))

        match = _CENTURY_ARABIC.search(era)
        if match:
            value = int(match.group(1))
            return -value if is_bce else value

        match = _CENTURY_ROMAN.search(era)
        if match:
            value = _ROMAN_VALUES.get(match.group(1).upper())
            if value:
                return -value if is_bce else value

        # Nessun secolo esplicito: si prova con un anno a quattro cifre.
        year_match = re.search(r"\b(\d{3,4})\b", era)
        if year_match:
            year = int(year_match.group(1))
            century = (year - 1) // 100 + 1
            return -century if is_bce else century

        return None

    @staticmethod
    def _extract_centuries(text: str) -> List[int]:
        """Secoli desunti dagli anni citati nel testo."""
        centuries: List[int] = []
        for match in _YEAR.finditer(text):
            try:
                year = int(match.group(1))
            except ValueError:
                continue
            if not 1 <= year <= 2100:
                continue

            suffix = (match.group(2) or "").lower().replace(" ", "").rstrip(".")
            is_bce = suffix in _BCE_MARKERS

            century = (year - 1) // 100 + 1
            centuries.append(-century if is_bce else century)

        return centuries


def filter_documents(
    documents: Sequence,
    author_name: str,
    era: str = "",
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[List, List[Tuple[str, RelevanceVerdict]]]:
    """Separa i documenti pertinenti da quelli scartati.

    Restituisce `(tenuti, scartati)`; gli scartati portano con se' il motivo,
    cosi' la decisione resta ispezionabile invece di sparire in silenzio.
    """
    relevance = RelevanceFilter(author_name, era, threshold)

    kept: List = []
    rejected: List[Tuple[str, RelevanceVerdict]] = []

    for document in documents:
        verdict = relevance.evaluate(
            document.title, document.text, document.is_primary_source
        )
        document.metadata["relevance_score"] = verdict.score
        if verdict.keep:
            kept.append(document)
        else:
            rejected.append((document.title, verdict))
            logger.info(
                f"Scartato '{document.title}' "
                f"(pertinenza {verdict.score:.2f}: {', '.join(verdict.reasons)})"
            )

    return kept, rejected
