"""Analisi della voce: come il personaggio *parla*, non cosa dice.

Questa dimensione mancava del tutto. Sintassi e vocabolario dicono quanto
sono lunghi i periodi e quali parole ricorrono, ma non catturano i tratti che
rendono una voce riconoscibile alla prima riga:

- gli **incipit**: con quali parole apre una frase (`Ita`, `Quibus rebus`,
  `Del resto`) — e' la firma ritmica piu' stabile di un autore;
- le **formule fisse**: le sequenze che ripete identiche;
- i **vocativi**: come si rivolge a chi ascolta;
- il **registro**: enfasi, domande, esclamazioni, incisi;
- la **posizione di chi parla**: quanto asserisce, quanto dubita, quanto ordina.

Tutto e' misurato per mille parole, cosi' i corpora di lunghezza diversa
restano confrontabili.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List

from ...data_models import SupportedLanguage, TextSegment, VoiceProfile
from ...stage1_ingestion.chunker import split_sentences

logger = logging.getLogger(__name__)

TOP_OPENERS = 25
TOP_VOCATIVES = 15

# Sotto questa soglia il corpus non basta a stabilire quali aperture siano
# caratteristiche dell'autore. Su un campione minuscolo qualunque formula
# ricorre in una frazione alta delle frasi per puro effetto della dimensione:
# dichiararla "tratto proprio" la impone a chi legge il profilo, e un modello
# addestrato su quel materiale impara un tic. Meglio non affermare nulla.
MIN_SENTENCES_FOR_SIGNATURE = 120

# Una apertura e' un tratto solo se ricorre senza dominare: comparire in oltre
# meta' delle frasi indica piu' un corpus omogeneo che uno stile.
MAX_OPENER_SHARE = 0.5

# Parole con cui si apre una frase: se ne prendono fino a due.
_LEADING_WORDS = re.compile(r"^[\"'«(\s]*([\w'’-]+)(?:\s+([\w'’-]+))?", re.UNICODE)

# Vocativo: nome proprio o appellativo fra virgole o dopo un'interiezione.
_VOCATIVE = re.compile(
    r"(?:^|[,;]\s*)(?:o\s+|oh\s+)?([A-ZÀ-ÖØ-Þ][\w'’-]{2,})\s*[,!]",
    re.UNICODE,
)

# Modalizzatori: segnalano se l'autore asserisce, dubita o comanda.
_MODALITY_MARKERS: Dict[str, Dict[str, List[str]]] = {
    "certainty": {
        "it": ["certamente", "senza dubbio", "indubbiamente", "è certo", "sicuramente"],
        "en": ["certainly", "without doubt", "undoubtedly", "surely", "it is certain"],
        "la": ["certe", "profecto", "sane", "nimirum", "scilicet", "haud dubie"],
        "grc": ["δηπου", "σαφως", "ακριβως", "οντως"],
        "fr": ["certainement", "sans doute", "assurément"],
        "de": ["gewiss", "zweifellos", "sicherlich"],
        "es": ["ciertamente", "sin duda", "seguramente"],
        "ru": ["конечно", "несомненно", "безусловно"],
        "hu": ["bizonyára", "kétségtelenül", "biztosan"],
    },
    "doubt": {
        "it": ["forse", "probabilmente", "sembra che", "pare che", "si direbbe"],
        "en": ["perhaps", "probably", "it seems", "apparently", "presumably"],
        "la": ["fortasse", "forsitan", "videtur", "credo", "opinor", "puto"],
        "grc": ["ισως", "δοκει", "ταχα", "που"],
        "fr": ["peut-être", "probablement", "il semble"],
        "de": ["vielleicht", "wahrscheinlich", "es scheint"],
        "es": ["quizás", "probablemente", "parece que"],
        "ru": ["возможно", "вероятно", "кажется"],
        "hu": ["talán", "valószínűleg", "úgy tűnik"],
    },
    "obligation": {
        "it": ["bisogna", "occorre", "si deve", "è necessario", "dovere"],
        "en": ["must", "ought to", "should", "it is necessary", "one has to"],
        "la": ["oportet", "necesse est", "debet", "opus est", "gerundum"],
        "grc": ["δει", "χρη", "αναγκη"],
        "fr": ["il faut", "on doit", "il est nécessaire"],
        "de": ["muss", "soll", "es ist nötig"],
        "es": ["hay que", "se debe", "es necesario"],
        "ru": ["нужно", "необходимо", "следует"],
        "hu": ["kell", "szükséges"],
    },
}


class VoiceAnalyzer:
    """Estrae i tratti di voce dal corpus."""

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.min_opener_count = config.get("analysis", {}).get("min_opener_count", 3)

    def analyze(self, segments: List[TextSegment]) -> VoiceProfile:
        if not segments:
            return VoiceProfile()

        openers: Counter[str] = Counter()
        vocatives: Counter[str] = Counter()
        modality: Counter[str] = Counter()

        total_words = 0
        total_sentences = 0
        questions = 0
        exclamations = 0
        parentheticals = 0
        direct_speech = 0
        word_lengths: List[int] = []

        languages = {seg.language.value for seg in segments}
        modality_patterns = self._compile_modality(languages)

        for segment in segments:
            content = segment.content
            total_words += segment.word_count

            for word in content.split():
                stripped = word.strip(".,;:!?\"'()[]")
                if stripped:
                    word_lengths.append(len(stripped))

            for sentence in split_sentences(content):
                total_sentences += 1

                match = _LEADING_WORDS.match(sentence)
                if match:
                    first = (match.group(1) or "").lower()
                    second = (match.group(2) or "").lower()
                    if first and len(first) > 1:
                        openers[first] += 1
                        if second and len(second) > 1:
                            openers[f"{first} {second}"] += 1

                if sentence.rstrip().endswith("?"):
                    questions += 1
                if sentence.rstrip().endswith("!"):
                    exclamations += 1

            for name in _VOCATIVE.findall(content):
                vocatives[name] += 1

            parentheticals += content.count("(") + content.count("—")
            direct_speech += content.count('"') // 2

            lowered = content.lower()
            for category, patterns in modality_patterns.items():
                for pattern in patterns:
                    modality[category] += len(pattern.findall(lowered))

        def per_1000(value: int) -> float:
            return round(value / total_words * 1000, 3) if total_words else 0.0

        signature_openers = self._signature_openers(openers, total_sentences, per_1000)

        total_modality = sum(modality.values())

        return VoiceProfile(
            signature_openers=signature_openers,
            vocatives=[
                {"term": term, "count": count}
                for term, count in vocatives.most_common(TOP_VOCATIVES)
                if count >= self.min_opener_count
            ],
            question_ratio=round(questions / total_sentences, 4) if total_sentences else 0.0,
            exclamation_ratio=round(exclamations / total_sentences, 4) if total_sentences else 0.0,
            parenthetical_per_1000_words=per_1000(parentheticals),
            direct_speech_per_1000_words=per_1000(direct_speech),
            avg_word_length=(
                round(sum(word_lengths) / len(word_lengths), 2) if word_lengths else 0.0
            ),
            modality_distribution=(
                {k: round(v / total_modality * 100, 2) for k, v in modality.items()}
                if total_modality else {}
            ),
            speech_register=self._infer_register(word_lengths, questions, total_sentences),
        )

    # ------------------------------------------------------------ supporto

    def _signature_openers(
        self, openers: Counter, total_sentences: int, per_1000
    ) -> List[Dict[str, Any]]:
        """Aperture caratteristiche, se il corpus basta a stabilirle.

        Due condizioni, entrambe necessarie:

        1. il corpus deve avere abbastanza frasi. Su un campione minuscolo
           ogni formula ricorre in una frazione alta delle frasi per effetto
           della dimensione, non dello stile;
        2. nessuna apertura deve dominare. Una formula presente in piu' di
           meta' delle frasi descrive un corpus omogeneo — poche pagine dello
           stesso passo — non un tratto d'autore.

        Quando le condizioni non sono soddisfatte non si riporta nulla: un
        profilo che tace su un punto e' piu' utile di uno che afferma il falso
        con sicurezza, perche' chi lo legge — un modello compreso — lo prende
        alla lettera.
        """
        if total_sentences < MIN_SENTENCES_FOR_SIGNATURE:
            logger.info(
                f"Corpus di {total_sentences} frasi: troppo poco per stabilire "
                f"le aperture caratteristiche (ne servono {MIN_SENTENCES_FOR_SIGNATURE}). "
                "Il profilo non ne riportera'."
            )
            return []

        selected: List[Dict[str, Any]] = []
        for opener, count in openers.most_common(TOP_OPENERS * 3):
            if count < self.min_opener_count:
                continue
            share = count / total_sentences
            if share > MAX_OPENER_SHARE:
                logger.info(
                    f"Apertura «{opener}» presente nel {share:.0%} delle frasi: "
                    "e' una caratteristica del corpus, non dell'autore. Esclusa."
                )
                continue
            selected.append({
                "opener": opener,
                "count": count,
                "share": round(share, 4),
                "per_1000_words": per_1000(count),
            })
            if len(selected) >= TOP_OPENERS:
                break

        return selected

    @staticmethod
    def _compile_modality(languages: set[str]) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for category, by_language in _MODALITY_MARKERS.items():
            patterns: List[re.Pattern] = []
            for lang_code, markers in by_language.items():
                if lang_code not in languages:
                    continue
                for marker in markers:
                    escaped = r"\s+".join(re.escape(p) for p in marker.split())
                    patterns.append(re.compile(rf"\b{escaped}\w{{0,3}}\b", re.IGNORECASE))
            if patterns:
                compiled[category] = patterns
        return compiled

    @staticmethod
    def _infer_register(
        word_lengths: List[int], questions: int, total_sentences: int
    ) -> str:
        """Registro stimato da lunghezza media delle parole e tasso di domande.

        Euristica dichiarata: parole lunghe indicano lessico dotto, molte
        domande indicano andamento dialogico. Non e' una misura assoluta,
        serve a orientare il tono del prompt di sistema.
        """
        if not word_lengths:
            return "unknown"

        avg = sum(word_lengths) / len(word_lengths)
        question_ratio = questions / total_sentences if total_sentences else 0.0

        if avg >= 6.0 and question_ratio < 0.05:
            return "formal_expository"
        if avg >= 6.0:
            return "formal_dialogic"
        if question_ratio >= 0.12:
            return "conversational_dialogic"
        return "plain_narrative"
