"""Analisi del vocabolario: quali parole sono davvero caratteristiche dell'autore.

Tre difetti della versione precedente:

1. un ciclo percorreva tutti i segmenti chiamando `extract_vocabulary` e poi
   faceva `pass`, raddoppiando la parte piu' lenta della pipeline per nulla;
2. il testo veniva concatenato in un'unica stringa per lingua e passato a
   spaCy con `max_length` alzato a piacere: su un libro sono decine di GB;
3. `unique_lemmas` veniva sommato fra lingue, ma la somma degli unici non e'
   il numero di unici, quindi type/token ratio e hapax erano sbagliati.

Sul merito: ordinare i lemmi per frequenza grezza restituisce *e, di, che* in
qualunque corpus. Un termine e' caratteristico se e' frequente **e** diffuso
lungo tutta l'opera: la frequenza da sola premia un termine ripetuto in una
sola pagina. Qui il punteggio combina le due cose.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from ...data_models import SupportedLanguage, TextSegment, VocabularyProfile
from ..nlp_manager import NLPManager
from ..stopwords import get_stopwords

logger = logging.getLogger(__name__)

# Numero di termini riportati nel profilo.
TOP_DISTINCTIVE = 60
TOP_COLLOCATIONS = 40

# Lunghezza minima di un lemma perche' sia considerato portatore di contenuto.
MIN_LEMMA_LENGTH = 3

# Sotto questa soglia il corpus non consente di distinguere una formula
# d'autore da una ripetizione casuale del campione. Dichiararla comunque la
# fa entrare nel profilo come tratto prescrittivo: un modello addestrato su
# quel materiale la ripete a ogni risposta.
MIN_TOKENS_FOR_FORMULAS = 5_000


class VocabularyAnalyzer:
    def __init__(self, nlp_manager: NLPManager, config: Dict[str, Any] | None = None):
        self.nlp_manager = nlp_manager
        config = config or {}
        analysis = config.get("analysis", {})
        self.max_chars_per_batch = analysis.get("max_chars_per_nlp_batch", 400_000)
        self.max_words = analysis.get("max_words_analyzed", 2_000_000)
        # Il nome del personaggio serve a escluderlo dalle formule: in un
        # testo che parla di lui e' la sequenza piu' ripetuta in assoluto.
        self.author_tokens = {
            part.lower()
            for part in str(config.get("persona", {}).get("author_name", "")).split()
            if len(part) > 2
        }

    def analyze(self, segments: List[TextSegment]) -> VocabularyProfile:
        by_language: Dict[SupportedLanguage, List[TextSegment]] = defaultdict(list)
        for segment in segments:
            by_language[segment.language].append(segment)

        # Frequenza globale e dispersione: in quanti segmenti distinti compare
        # ogni lemma. Un lemma frequente ma concentrato non e' caratteristico.
        global_freq: Counter[str] = Counter()
        segment_presence: Counter[str] = Counter()
        total_tokens = 0
        unique_lemmas: set[str] = set()
        total_segments = 0
        bigram_freq: Counter[Tuple[str, ...]] = Counter()
        trigram_freq: Counter[Tuple[str, ...]] = Counter()

        all_stopwords: set[str] = set()

        for language, lang_segments in by_language.items():
            if language == SupportedLanguage.UNKNOWN:
                continue

            parser = self.nlp_manager.get_parser(language)
            stopwords = get_stopwords(language, parser)
            all_stopwords |= stopwords

            for segment, lemmas in self._iter_segment_lemmas(parser, lang_segments):
                total_segments += 1
                total_tokens += len(lemmas)

                content_lemmas = [
                    lemma for lemma in lemmas
                    if self._is_content_lemma(lemma, stopwords)
                ]

                global_freq.update(content_lemmas)
                unique_lemmas.update(lemmas)
                segment_presence.update(set(content_lemmas))

                # Le collocazioni si calcolano sulla sequenza completa: sono
                # formule ("qua de causa"), quindi le stopword ne fanno parte.
                bigram_freq.update(zip(lemmas, lemmas[1:]))
                trigram_freq.update(zip(lemmas, lemmas[1:], lemmas[2:]))

        if total_tokens == 0:
            logger.warning("Nessun token analizzabile: profilo vocabolario vuoto")
            return VocabularyProfile(
                distinctive_terms=[], semantic_field_distribution={},
                hapax_legomena_ratio=0.0, type_token_ratio=0.0,
                formulas_and_collocations=[],
            )

        hapax_count = sum(1 for count in global_freq.values() if count == 1)

        distinctive = self._rank_distinctive(
            global_freq, segment_presence, total_tokens, max(total_segments, 1)
        )
        collocations = (
            self._rank_collocations(
                bigram_freq, trigram_freq, total_segments,
                self.author_tokens, all_stopwords,
            )
            if total_tokens >= MIN_TOKENS_FOR_FORMULAS
            else []
        )
        if not collocations and total_tokens < MIN_TOKENS_FOR_FORMULAS:
            logger.info(
                f"Corpus di {total_tokens} token: troppo poco per distinguere le "
                f"formule d'autore dalle ripetizioni casuali "
                f"(ne servono {MIN_TOKENS_FOR_FORMULAS}). Il profilo non ne riportera'."
            )

        return VocabularyProfile(
            distinctive_terms=distinctive,
            semantic_field_distribution=self._semantic_fields(global_freq, by_language),
            hapax_legomena_ratio=round(hapax_count / total_tokens, 4),
            type_token_ratio=round(len(unique_lemmas) / total_tokens, 4),
            formulas_and_collocations=collocations,
        )

    # --------------------------------------------------------------- lemmi

    def _iter_segment_lemmas(
        self, parser, segments: List[TextSegment]
    ) -> Iterable[Tuple[TextSegment, List[str]]]:
        """Produce i lemmi segmento per segmento, a lotti di memoria limitata.

        Ogni segmento viene analizzato una volta sola: il risultato e' usato
        subito, non ricalcolato come accadeva prima.
        """
        words_seen = 0
        batch: List[TextSegment] = []
        batch_chars = 0

        def flush(current: List[TextSegment]):
            if not current:
                return []
            texts = [s.content for s in current]
            try:
                return list(parser.extract_lemmas_batch(texts))
            except Exception as exc:
                logger.warning(f"Estrazione lemmi fallita su un lotto: {exc}")
                return [[] for _ in current]

        for segment in segments:
            if words_seen >= self.max_words:
                logger.info(
                    f"Limite di {self.max_words} parole raggiunto: "
                    "analisi del vocabolario troncata (analysis.max_words_analyzed)"
                )
                break

            if batch and batch_chars + segment.char_count > self.max_chars_per_batch:
                for seg, lemmas in zip(batch, flush(batch)):
                    yield seg, lemmas
                batch, batch_chars = [], 0

            batch.append(segment)
            batch_chars += segment.char_count
            words_seen += segment.word_count

        for seg, lemmas in zip(batch, flush(batch)):
            yield seg, lemmas

    @staticmethod
    def _is_content_lemma(lemma: str, stopwords: set) -> bool:
        """Vero se il lemma porta contenuto.

        spaCy scompone le preposizioni articolate: «del» diventa il lemma
        «di il». Il risultato non compare in nessuna lista di stopword, cosi'
        le preposizioni piu' frequenti della lingua finivano in cima ai
        termini caratteristici — su un corpus italiano il lessico distintivo
        risultava «di il, a il, in il, da il».
        """
        if len(lemma) < MIN_LEMMA_LENGTH or lemma.isdigit():
            return False
        if lemma in stopwords:
            return False
        # Lemma composto: e' contenuto solo se lo e' ogni sua parte.
        if " " in lemma:
            return all(
                part not in stopwords and len(part) >= MIN_LEMMA_LENGTH
                for part in lemma.split()
            )
        return True

    # ------------------------------------------------------------ punteggi

    @staticmethod
    def _rank_distinctive(
        freq: Counter[str],
        presence: Counter[str],
        total_tokens: int,
        total_segments: int,
    ) -> List[Dict[str, Any]]:
        """Ordina i lemmi per frequenza *ponderata dalla diffusione*.

        `log(1 + f)` smorza la frequenza — senza, il termine piu' ripetuto
        domina qualsiasi classifica — e la moltiplicazione per la quota di
        segmenti in cui compare privilegia cio' che attraversa tutta l'opera.
        """
        scored: List[Dict[str, Any]] = []
        for lemma, count in freq.items():
            if count < 2:
                continue
            dispersion = presence[lemma] / total_segments
            score = math.log1p(count) * dispersion
            scored.append({
                "term": lemma,
                "count": count,
                "freq": round(count / total_tokens, 6),
                "dispersion": round(dispersion, 4),
                "score": round(score, 4),
            })

        scored.sort(key=lambda item: -item["score"])
        return scored[:TOP_DISTINCTIVE]

    @staticmethod
    def _rank_collocations(
        bigrams: Counter[Tuple[str, ...]],
        trigrams: Counter[Tuple[str, ...]],
        total_segments: int,
        author_tokens: set | None = None,
        stopwords: set | None = None,
    ) -> List[str]:
        """Sequenze ricorrenti: le formule che l'autore ripete.

        Le sequenze che contengono il nome del personaggio sono escluse: in
        un testo che parla di lui — una biografia, una voce enciclopedica —
        il suo nome e' la sequenza piu' ripetuta, e le "formule d'autore"
        risultavano «lucio anneo seneca», «seneca il vecchio».

        La soglia sale col numero di segmenti: su un corpus grande due
        occorrenze sono casualita', su uno piccolo sono un tratto.
        """
        min_count = max(3, total_segments // 50)

        author_tokens = author_tokens or set()
        stopwords = stopwords or set()

        def names_the_author(ngram) -> bool:
            return any(word.lower() in author_tokens for word in ngram)

        def is_junk(ngram) -> bool:
            """Sequenze che non sono formule d'autore.

            Tre artefatti ricorrenti: la ripetizione dello stesso lemma
            («essere essere»), prodotta dalla lemmatizzazione di verbi
            composti; le sequenze fatte di sole parole funzionali («di il
            suo»), che sono grammatica della lingua e non scelta di stile; e
            il boilerplate della fonte («istituto di il enciclopedia
            italiana»), che appartiene a chi ha pubblicato il testo.
            """
            if any(a == b for a, b in zip(ngram, ngram[1:])):
                return True
            content = [w for w in ngram if w.lower() not in stopwords and len(w) > 2]
            return len(content) < 2

        formulas: List[Tuple[int, str]] = []
        # I trigrammi valgono piu' dei bigrammi a parita' di occorrenze:
        # sono formule piu' specifiche e meno probabili per caso.
        for ngram, count in trigrams.items():
            if (count >= min_count and all(len(w) > 1 for w in ngram)
                    and not names_the_author(ngram) and not is_junk(ngram)):
                formulas.append((count * 2, " ".join(ngram)))
        for ngram, count in bigrams.items():
            if (count >= min_count * 2 and all(len(w) > 2 for w in ngram)
                    and not names_the_author(ngram) and not is_junk(ngram)):
                formulas.append((count, " ".join(ngram)))

        formulas.sort(key=lambda item: -item[0])

        seen: set[str] = set()
        result: List[str] = []
        for _, phrase in formulas:
            # Un bigramma gia' contenuto in un trigramma accettato e' ridondante.
            if any(phrase in kept for kept in seen):
                continue
            seen.add(phrase)
            result.append(phrase)
            if len(result) >= TOP_COLLOCATIONS:
                break
        return result

    # ------------------------------------------------------- campi semantici

    @staticmethod
    def _semantic_fields(
        freq: Counter[str],
        by_language: Dict[SupportedLanguage, List[TextSegment]],
    ) -> Dict[str, float]:
        """Quota di occorrenze per campo semantico, dai dizionari su disco."""
        from ..semantic_fields import load_semantic_fields

        fields = load_semantic_fields()
        if not fields:
            return {}

        totals: Dict[str, int] = {}
        for field_name, terms in fields.items():
            hits = 0
            for term in terms:
                term_lower = term.lower()
                hits += freq.get(term_lower, 0)
                # I dizionari elencano forme piene; i lemmi possono differire
                # nella coda, quindi si accetta anche la corrispondenza per radice.
                if len(term_lower) >= 5:
                    stem = term_lower[:-2]
                    hits += sum(
                        count for lemma, count in freq.items()
                        if lemma != term_lower and lemma.startswith(stem)
                    )
            if hits:
                totals[field_name] = hits

        grand_total = sum(totals.values())
        if not grand_total:
            return {}
        return {k: round(v / grand_total * 100, 2) for k, v in totals.items()}
