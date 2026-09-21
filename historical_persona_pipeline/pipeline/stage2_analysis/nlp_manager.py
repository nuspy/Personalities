"""Gestione dei modelli linguistici, con caricamento pigro per lingua.

Interfaccia comune ai tre backend (spaCy, CLTK, fallback regex):

    analyze_syntax_batch(texts)  -> lista di dict di statistiche sintattiche
    extract_lemmas_batch(texts)  -> lista di liste di lemmi
    stopwords                    -> set di stopword del modello, se esiste

L'elaborazione e' sempre a lotti: `nlp.pipe` e' molto piu' veloce di una
chiamata per testo, e alzare `max_length` per far passare un intero libro in
una sola chiamata — come faceva la versione precedente — e' proprio cio' che
fa esplodere la memoria.
"""
from __future__ import annotations

import gc
import logging
import re
from typing import Any, Dict, Iterable, List, Sequence

from ..data_models import SupportedLanguage

logger = logging.getLogger(__name__)

# Dipendenze che segnalano una subordinata.
SUBORDINATE_DEPS = frozenset({
    "ccomp", "xcomp", "advcl", "acl", "relcl", "acl:relcl", "csubj", "csubjpass",
})
COORDINATE_DEPS = frozenset({"conj", "cc"})

# Oltre questa soglia spaCy rischia di saturare la memoria: i testi piu' lunghi
# vengono spezzati prima di entrare nella pipeline.
SPACY_SAFE_CHARS = 100_000


class NLPManager:
    SPACY_MODELS = {
        SupportedLanguage.ITALIAN: "it_core_news_lg",
        SupportedLanguage.ENGLISH: "en_core_web_lg",
        SupportedLanguage.FRENCH: "fr_core_news_lg",
        SupportedLanguage.GERMAN: "de_core_news_lg",
        SupportedLanguage.SPANISH: "es_core_news_lg",
        SupportedLanguage.PORTUGUESE: "pt_core_news_lg",
        SupportedLanguage.RUSSIAN: "ru_core_news_lg",
        SupportedLanguage.UKRAINIAN: "uk_core_news_lg",
        SupportedLanguage.GREEK_MODERN: "el_core_news_lg",
        SupportedLanguage.HUNGARIAN: "hu_core_news_trf",
    }

    # Modelli piccoli: ripiego automatico quando il modello grande non c'e'.
    SPACY_FALLBACK_MODELS = {
        SupportedLanguage.ITALIAN: "it_core_news_sm",
        SupportedLanguage.ENGLISH: "en_core_web_sm",
        SupportedLanguage.FRENCH: "fr_core_news_sm",
        SupportedLanguage.GERMAN: "de_core_news_sm",
        SupportedLanguage.SPANISH: "es_core_news_sm",
        SupportedLanguage.PORTUGUESE: "pt_core_news_sm",
        SupportedLanguage.RUSSIAN: "ru_core_news_sm",
        SupportedLanguage.UKRAINIAN: "uk_core_news_sm",
        SupportedLanguage.GREEK_MODERN: "el_core_news_sm",
        SupportedLanguage.HUNGARIAN: "hu_core_news_md",
    }

    CLTK_LANGUAGES = {
        SupportedLanguage.GREEK_ANCIENT: "grc",
        SupportedLanguage.LATIN: "lat",
    }

    def __init__(self, auto_download: bool = True):
        self._loaded_models: Dict[SupportedLanguage, Any] = {}
        self.auto_download = auto_download

    def get_parser(self, language: SupportedLanguage):
        if language not in self._loaded_models:
            self._loaded_models[language] = self._load_parser(language)
        return self._loaded_models[language]

    def _load_parser(self, language: SupportedLanguage):
        logger.info(f"Caricamento modello NLP per {language.value}...")
        try:
            if language in self.CLTK_LANGUAGES:
                return self._load_cltk_parser(language)
            if language in self.SPACY_MODELS:
                return self._load_spacy_parser(language)
        except Exception as exc:
            logger.warning(
                f"Modello per {language.value} non disponibile ({exc}): "
                "uso l'analizzatore di base (statistiche ridotte)."
            )
        return BasicParser(language)

    def _load_spacy_parser(self, language: SupportedLanguage):
        import spacy

        large = self.SPACY_MODELS[language]
        small = self.SPACY_FALLBACK_MODELS.get(language)

        # Prima si prova a caricare cio' che e' gia' installato, grande o
        # piccolo che sia: nessun download se il modello c'e'.
        for model_name in (large, small):
            if not model_name:
                continue
            try:
                return SpacyWrapper(spacy.load(model_name), language)
            except OSError:
                continue

        if not self.auto_download:
            raise OSError(
                f"Nessun modello spaCy installato per {language.value}. "
                f"Installarlo con: python -m spacy download {small or large}"
            )

        # Si scarica il modello **piccolo**: qualche decina di MB contro le
        # centinaia del modello grande. In una pipeline non presidiata un
        # download da 550 MB deciso in autonomia non e' accettabile; chi
        # vuole il modello grande lo installa esplicitamente.
        target = small or large
        logger.warning(
            f"Nessun modello per {language.value}. Scarico '{target}' "
            "(alcune decine di MB). Per una qualita' superiore installare "
            f"manualmente '{large}'."
        )
        if self._download_spacy_model(target):
            try:
                return SpacyWrapper(spacy.load(target), language)
            except OSError:
                pass

        raise OSError(f"Nessun modello spaCy disponibile per {language.value}")

    @staticmethod
    def _download_spacy_model(model_name: str) -> bool:
        import subprocess
        import sys

        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True, capture_output=True, timeout=1800,
            )
            return True
        except Exception as exc:
            logger.warning(f"Download di {model_name} fallito: {exc}")
            return False

    def _load_cltk_parser(self, language: SupportedLanguage):
        from cltk import NLP

        lang_code = self.CLTK_LANGUAGES[language]
        return CLTKWrapper(NLP(language=lang_code, suppress_banner=True), language)

    def unload(self, language: SupportedLanguage | None = None) -> None:
        if language is not None:
            self._loaded_models.pop(language, None)
        else:
            self._loaded_models.clear()
        gc.collect()


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

def _split_oversized(texts: Sequence[str], limit: int) -> List[List[str]]:
    """Spezza i testi troppo lunghi, ricordando come ricomporli."""
    groups: List[List[str]] = []
    for text in texts:
        if len(text) <= limit:
            groups.append([text])
        else:
            groups.append([text[i:i + limit] for i in range(0, len(text), limit)])
    return groups


class BasicParser:
    """Fallback senza modello: statistiche ottenute con espressioni regolari."""

    _SENTENCE = re.compile(r"[.!?;]+")
    _WORD = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(self, language: SupportedLanguage):
        self.language = language
        self.stopwords: set[str] = set()

    def analyze_syntax_batch(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        return [self._analyze_one(t) for t in texts]

    def _analyze_one(self, text: str) -> Dict[str, Any]:
        sentences = [s.strip() for s in self._SENTENCE.split(text) if s.strip()]
        return {
            "sentences": [{"length": len(s.split())} for s in sentences],
            "person_counts": {},       # senza POS tagger non e' determinabile
            "pos_sequences": [],
            "has_morphology": False,
        }

    def extract_lemmas_batch(self, texts: Sequence[str]) -> List[List[str]]:
        # Senza lemmatizzatore si usano le forme minuscole: meno preciso sulle
        # lingue flessive, ma sufficiente per frequenza e collocazioni.
        return [self._WORD.findall(t.lower()) for t in texts]


class SpacyWrapper:
    def __init__(self, nlp, language: SupportedLanguage):
        self.nlp = nlp
        self.language = language
        self.stopwords = set(getattr(nlp.Defaults, "stop_words", set()))
        # Tetto di sicurezza fisso: l'input viene spezzato prima di arrivare qui.
        self.nlp.max_length = max(self.nlp.max_length, SPACY_SAFE_CHARS + 10_000)

    def _pipe(self, texts: Sequence[str], disable: Sequence[str] = ()):
        groups = _split_oversized(texts, SPACY_SAFE_CHARS)
        flat = [piece for group in groups for piece in group]
        docs = list(self.nlp.pipe(flat, disable=list(disable), batch_size=16))

        # Ricompone i pezzi nello stesso ordine dei testi in ingresso.
        out, cursor = [], 0
        for group in groups:
            out.append(docs[cursor:cursor + len(group)])
            cursor += len(group)
        return out

    def analyze_syntax_batch(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for doc_group in self._pipe(texts):
            sentences: List[Dict[str, Any]] = []
            person_counts: Dict[str, int] = {"1": 0, "2": 0, "3": 0}
            pos_sequences: List[tuple] = []

            for doc in doc_group:
                for sent in doc.sents:
                    tokens = [t for t in sent if not t.is_punct and not t.is_space]
                    if not tokens:
                        continue

                    deps = {t.dep_ for t in sent}
                    sentences.append({
                        "length": len(tokens),
                        "has_subordinate": bool(deps & SUBORDINATE_DEPS),
                        "has_coordination": bool(deps & COORDINATE_DEPS),
                        "clause_count": sum(1 for t in sent if t.pos_ in ("VERB", "AUX")),
                        "is_question": sent.text.strip().endswith(("?", ";")),
                    })

                    pos_sequences.append(tuple(t.pos_ for t in tokens[:6]))

                    for token in sent:
                        if token.pos_ in ("VERB", "AUX"):
                            person = token.morph.get("Person")
                            if person and person[0] in person_counts:
                                person_counts[person[0]] += 1

            results.append({
                "sentences": sentences,
                "person_counts": person_counts,
                "pos_sequences": pos_sequences,
                "has_morphology": True,
            })

        return results

    def extract_lemmas_batch(self, texts: Sequence[str]) -> List[List[str]]:
        results: List[List[str]] = []
        # parser e NER non servono ai lemmi: disattivarli dimezza il tempo.
        for doc_group in self._pipe(texts, disable=("parser", "ner")):
            lemmas: List[str] = []
            for doc in doc_group:
                lemmas.extend(
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_punct and not token.is_space
                )
            results.append(lemmas)
        return results


class CLTKWrapper:
    """Backend per latino e greco antico.

    CLTK e' lento e la sua API varia fra versioni: ogni accesso ai tratti
    morfologici e' difensivo e, se l'analisi fallisce, si ripiega su
    `BasicParser` invece di perdere il segmento.
    """

    def __init__(self, cltk_nlp, language: SupportedLanguage):
        self.nlp = cltk_nlp
        self.language = language
        self.stopwords: set[str] = set()
        self._basic = BasicParser(language)

    def analyze_syntax_batch(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        return [self._analyze_one(t) for t in texts]

    def _analyze_one(self, text: str) -> Dict[str, Any]:
        try:
            doc = self.nlp.analyze(text)
        except Exception as exc:
            logger.debug(f"Analisi CLTK fallita, uso il fallback: {exc}")
            return self._basic._analyze_one(text)

        sentences: List[Dict[str, Any]] = []
        person_counts: Dict[str, int] = {"1": 0, "2": 0, "3": 0}

        try:
            for sent in doc.sentences:
                words = list(sent.words)
                sentences.append({
                    "length": len(words),
                    "clause_count": sum(
                        1 for w in words if str(getattr(w, "pos", "")).lower() == "verb"
                    ),
                })
                for word in words:
                    if str(getattr(word, "pos", "")).lower() != "verb":
                        continue
                    features = getattr(word, "features", None)
                    if not features:
                        continue
                    try:
                        person = features["Person"]
                    except Exception:
                        continue
                    digits = re.findall(r"[123]", str(person))
                    if digits and digits[0] in person_counts:
                        person_counts[digits[0]] += 1
        except Exception as exc:
            logger.debug(f"Estrazione tratti CLTK fallita: {exc}")
            return self._basic._analyze_one(text)

        return {
            "sentences": sentences,
            "person_counts": person_counts,
            "pos_sequences": [],
            "has_morphology": True,
        }

    def extract_lemmas_batch(self, texts: Sequence[str]) -> List[List[str]]:
        results: List[List[str]] = []
        for text in texts:
            try:
                doc = self.nlp.analyze(text)
                lemmas = [
                    str(w.lemma).lower()
                    for w in doc.words
                    if getattr(w, "lemma", None)
                ]
                results.append(lemmas or self._basic.extract_lemmas_batch([text])[0])
            except Exception:
                results.append(self._basic.extract_lemmas_batch([text])[0])
        return results
