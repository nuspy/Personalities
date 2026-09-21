"""Stage 1 — ingestione: da file eterogenei a segmenti di testo normalizzati.

Correzioni rispetto alla versione precedente:
- i decoder arrivano dal registro condiviso, quindi la GUI non puo' piu'
  offrire formati che la pipeline non sa leggere;
- un decoder che fallisce non interrompe l'intero lotto e l'errore viene
  riportato, invece di produrre un corpus vuoto in silenzio;
- `total_files_processed` conta i file davvero letti, non quelli selezionati;
- le sezioni troppo lunghe vengono suddivise, quelle troppo corte accorpate:
  il rilevamento lingua su frammenti di 40 caratteri era inaffidabile;
- un formato sconosciuto ma leggibile come testo non viene piu' scartato.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data_models import FileFormat, IngestionResult, SupportedLanguage, TextSegment
from ..stage_base import PipelineStage
from .chunker import chunk_text
from .decoders import build_decoders, find_decoder
from .decoders.txt_decoder import TXTDecoder
from .format_detector import FormatDetector
from .language_detector import LanguageDetector
from .text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

# Sotto questa lunghezza un segmento non porta informazione di stile
# utilizzabile e falsa il rilevamento di lingua.
MIN_SEGMENT_CHARS = 120


class IngestionStage(PipelineStage):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        ingestion_config = config.get("ingestion", {})
        self.target_chunk_chars = ingestion_config.get("target_chunk_chars", 4000)
        self.min_chunk_chars = ingestion_config.get("min_chunk_chars", 200)

        self.format_detector = FormatDetector()
        self.language_detector = LanguageDetector()
        self.normalizer = TextNormalizer()
        self.decoders = build_decoders(config)
        self._text_fallback = TXTDecoder(self.target_chunk_chars, self.min_chunk_chars)

    # ------------------------------------------------------------------ run

    def run(self, file_paths: List[Path]) -> IngestionResult:
        file_paths = [Path(p) for p in file_paths]
        total_files = len(file_paths)
        self.logger.info(f"Ingestione di {total_files} file")
        self.progress_update.emit(0, "Avvio ingestione...")

        segments: List[TextSegment] = []
        errors: List[Dict[str, Any]] = []
        language_counts: Dict[SupportedLanguage, int] = {}
        files_read = 0

        for idx, file_path in enumerate(file_paths):
            progress = int((idx / total_files) * 100) if total_files else 0
            self.progress_update.emit(progress, f"Lettura di {file_path.name}...")

            if not file_path.exists():
                errors.append({"file": str(file_path), "error": "File non trovato"})
                continue

            file_segments, file_errors, read_ok = self._process_file(file_path)
            errors.extend(file_errors)
            if read_ok:
                files_read += 1

            for segment in file_segments:
                language_counts[segment.language] = language_counts.get(segment.language, 0) + 1
            segments.extend(file_segments)

        self.progress_update.emit(
            100, f"Ingestione completata: {len(segments)} segmenti da {files_read} file"
        )

        if not segments:
            detail = "; ".join(e["error"] for e in errors[:3]) or "nessun testo estratto"
            raise ValueError(f"Ingestione senza risultato utile ({detail})")

        result = IngestionResult(
            project_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            total_files_processed=files_read,
            total_segments=len(segments),
            segments=segments,
            language_distribution=language_counts,
            errors=errors,
        )

        self.stage_completed.emit(result)
        return result

    # --------------------------------------------------------------- per file

    def _process_file(
        self, file_path: Path
    ) -> tuple[List[TextSegment], List[Dict[str, Any]], bool]:
        errors: List[Dict[str, Any]] = []

        file_fmt, _confidence = self.format_detector.detect(file_path)
        decoder = find_decoder(file_path, self.decoders)

        if decoder is None:
            # Formato non riconosciuto: se il contenuto e' testo leggibile lo
            # trattiamo come tale invece di scartare il file.
            if self._looks_like_text(file_path):
                decoder = self._text_fallback
                errors.append({
                    "file": str(file_path),
                    "error": f"Formato {file_fmt.value} non riconosciuto: letto come testo semplice",
                    "severity": "warning",
                })
            else:
                errors.append({
                    "file": str(file_path),
                    "error": f"Nessun decoder per il formato {file_fmt.value}",
                })
                return [], errors, False

        result = decoder.safe_decode(file_path)

        for message in result["errors"]:
            errors.append({"file": str(file_path), "error": message})

        items = result["sections"]
        if not items and result["text"].strip():
            items = [{"content": result["text"], "type": "full_text"}]

        if not items:
            return [], errors, False

        segments = self._build_segments(file_path, file_fmt, items)
        return segments, errors, True

    def _build_segments(
        self,
        file_path: Path,
        file_fmt: FileFormat,
        items: List[Dict[str, Any]],
    ) -> List[TextSegment]:
        segments: List[TextSegment] = []
        carry_over = ""  # frammento troppo corto in attesa di essere accorpato

        for item in items:
            content = (item.get("content") or "").strip()
            if not content:
                continue

            normalized = self.normalizer.normalize(content)
            if carry_over:
                normalized = f"{carry_over}\n\n{normalized}"
                carry_over = ""

            # Una sezione molto lunga (un capitolo intero, una pagina densa)
            # va suddivisa: alimentare spaCy con blocchi illimitati fa esplodere
            # la memoria sui libri.
            pieces = (
                chunk_text(normalized, self.target_chunk_chars, self.min_chunk_chars)
                if len(normalized) > self.target_chunk_chars * 2
                else [normalized]
            )

            for piece in pieces:
                if len(piece) < MIN_SEGMENT_CHARS:
                    carry_over = piece
                    continue

                language, confidence = self.language_detector.detect(piece)
                segments.append(TextSegment(
                    id=str(uuid.uuid4()),
                    content=piece,
                    language=language,
                    language_confidence=confidence,
                    source_file=file_path.name,
                    source_format=file_fmt,
                    page_or_section=self._section_label(item),
                    word_count=len(piece.split()),
                    char_count=len(piece),
                ))

        # L'ultimo frammento corto non va perso: si attacca al segmento precedente.
        if carry_over and segments:
            last = segments[-1]
            merged = f"{last.content}\n\n{carry_over}"
            segments[-1] = last.model_copy(update={
                "content": merged,
                "word_count": len(merged.split()),
                "char_count": len(merged),
            })

        return segments

    # ------------------------------------------------------------- utilities

    @staticmethod
    def _section_label(item: Dict[str, Any]) -> Optional[str]:
        for key in ("heading", "number", "type"):
            value = item.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _looks_like_text(file_path: Path, probe_bytes: int = 4096) -> bool:
        """Vero se i primi byte non contengono marcatori binari."""
        try:
            with open(file_path, "rb") as fh:
                sample = fh.read(probe_bytes)
        except OSError:
            return False
        if not sample:
            return False
        if b"\x00" in sample:
            return False
        # Molti byte di controllo => binario.
        control = sum(1 for b in sample if b < 9 or (13 < b < 32))
        return control / len(sample) < 0.05
