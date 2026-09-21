"""Suddivisione del testo in blocchi di dimensione governata.

Motivo: l'ingestione produceva un segmento per paragrafo (migliaia di
frammenti da 40 caratteri su cui il rilevamento lingua e' inaffidabile) e
l'analisi ricomponeva l'intero corpus in una stringa unica da dare a spaCy
(decine di GB di RAM su un libro). Entrambi gli estremi sono sbagliati.

Qui i paragrafi vengono accorpati in blocchi di circa `target_chars`
rispettando i confini naturali: mai una frase spezzata a meta'.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator, List

# Fine frase: punteggiatura terminale seguita da spazio e maiuscola/virgolette.
# Include il punto interrogativo greco (;) e quello arabo, utili sui corpora antichi.
_SENTENCE_END = re.compile(r'(?<=[.!?;؟])\s+(?=["\'«»(]?[A-ZÀ-ÖØ-ÞΑ-ΩΆ-ΏА-Я])')

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def split_sentences(text: str) -> List[str]:
    """Divisione in frasi indipendente dalla lingua (euristica, non NLP)."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def chunk_text(
    text: str,
    target_chars: int = 4000,
    min_chars: int = 200,
    max_chars: int | None = None,
) -> List[str]:
    """Accorpa il testo in blocchi di circa `target_chars` caratteri.

    - I paragrafi non vengono mai spezzati, salvo superino da soli `max_chars`;
      in quel caso si spezza sui confini di frase.
    - Un blocco finale piu' corto di `min_chars` viene fuso col precedente:
      frammenti brevi falsano il rilevamento lingua e le statistiche di stile.
    """
    if max_chars is None:
        max_chars = target_chars * 2

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append("\n\n".join(buffer))
            buffer = []
            buffer_len = 0

    for paragraph in split_paragraphs(text):
        if len(paragraph) > max_chars:
            flush()
            chunks.extend(_split_long_paragraph(paragraph, target_chars, max_chars))
            continue

        if buffer_len + len(paragraph) > target_chars and buffer:
            flush()

        buffer.append(paragraph)
        buffer_len += len(paragraph) + 2

    flush()

    # Coda troppo corta: la si attacca al blocco precedente.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()

    return [c for c in chunks if c.strip()]


def _split_long_paragraph(paragraph: str, target_chars: int, max_chars: int) -> List[str]:
    """Spezza un paragrafo monolitico sui confini di frase."""
    sentences = split_sentences(paragraph)
    if not sentences:
        # Nessun confine di frase riconoscibile (es. testo senza punteggiatura):
        # taglio netto, unico caso in cui accettiamo di spezzare arbitrariamente.
        return [paragraph[i:i + target_chars] for i in range(0, len(paragraph), target_chars)]

    out: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for sentence in sentences:
        if buffer_len + len(sentence) > target_chars and buffer:
            out.append(" ".join(buffer))
            buffer, buffer_len = [], 0
        if len(sentence) > max_chars:
            if buffer:
                out.append(" ".join(buffer))
                buffer, buffer_len = [], 0
            out.extend(sentence[i:i + target_chars] for i in range(0, len(sentence), target_chars))
            continue
        buffer.append(sentence)
        buffer_len += len(sentence) + 1

    if buffer:
        out.append(" ".join(buffer))
    return out


def batch_by_chars(texts: Iterable[str], max_chars: int) -> Iterator[List[str]]:
    """Raggruppa testi in lotti sotto `max_chars` complessivi.

    Serve all'analisi NLP: spaCy va alimentato a lotti con un tetto di memoria
    noto, invece che con l'intero corpus concatenato.
    """
    batch: List[str] = []
    size = 0
    for text in texts:
        length = len(text)
        if batch and size + length > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(text)
        size += length
    if batch:
        yield batch
