"""Stage 0 — ricerca: costruisce il corpus partendo dal solo nome.

E' il passo che rende la pipeline autonoma. Prima l'unico ingresso possibile
erano i file che l'utente selezionava a mano; con questo stadio bastano un
nome e un'epoca perche' il corpus si formi da solo, e i documenti caricati
manualmente restano un'aggiunta, non un requisito.

I testi scaricati sono salvati come file nella cartella di progetto: da li'
in poi attraversano la stessa ingestione dei documenti locali, quindi non
esiste un percorso privilegiato per il materiale online.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..stage_base import PipelineStage
from .relevance import filter_documents
from .sources import SOURCE_REGISTRY, ResearchDocument

logger = logging.getLogger(__name__)

# Caratteri non ammessi nei nomi file su Windows.
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class ResearchStage(PipelineStage):
    """Cerca, scarica e mette su disco il materiale su un personaggio."""

    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = Path(project_dir)
        research = config.get("research", {})
        self.enabled = research.get("enabled", True)
        self.max_documents = research.get("max_documents", 25)
        self.max_chars = research.get("max_chars_per_document", 400_000)
        self.source_names = research.get("sources", ["wikipedia", "wikisource", "gutenberg"])
        self.user_agent = research.get(
            "user_agent", "HistoricalPersonaPipeline/0.2 (local research tool)"
        )
        self.relevance_threshold = research.get("relevance_threshold", 0.40)

    def run(self, request: Dict[str, Any]) -> List[Path]:
        """Esegue la ricerca e restituisce i percorsi dei file scaricati.

        `request` accetta: `author_name` (obbligatorio), `era`, `languages`,
        `extra_queries`.
        """
        author = (request.get("author_name") or "").strip()
        if not author:
            raise ValueError("Ricerca online senza nome del personaggio")

        if not self.enabled:
            self.logger.info("Ricerca online disabilitata in configurazione")
            return []

        languages = request.get("languages") or self._default_languages(request.get("era", ""))
        queries = self._build_queries(author, request.get("era", ""), request.get("extra_queries"))

        self.progress_update.emit(0, f"Ricerca di materiale su {author}...")

        documents: List[ResearchDocument] = []
        seen_urls: set[str] = set()

        total_steps = max(1, len(self.source_names) * len(languages))
        step = 0

        for source_name in self.source_names:
            if len(documents) >= self.max_documents:
                break

            source_class = SOURCE_REGISTRY.get(source_name)
            if source_class is None:
                self.logger.warning(f"Fonte sconosciuta ignorata: {source_name}")
                continue

            source = source_class(self.user_agent, self.max_chars)

            for language in languages:
                if len(documents) >= self.max_documents:
                    break

                step += 1
                self.progress_update.emit(
                    int(step / total_steps * 85),
                    f"{source_name} [{language}]: ricerca in corso...",
                )

                for query in queries:
                    if len(documents) >= self.max_documents:
                        break
                    try:
                        found = source.search(query, language=language, limit=4)
                    except Exception as exc:
                        self.logger.warning(f"{source_name} fallita su '{query}': {exc}")
                        continue

                    for document in found:
                        if document.url in seen_urls:
                            continue
                        seen_urls.add(document.url)
                        documents.append(document)
                        if len(documents) >= self.max_documents:
                            break

        if not documents:
            self.error_occurred.emit(
                f"Nessun documento trovato online per '{author}'. "
                "Verificare la connessione o caricare documenti manualmente."
            )
            return []

        # Una ricerca per nome intercetta omonimi e parenti: senza questo
        # filtro il profilo misurerebbe la media fra persone diverse.
        self.progress_update.emit(88, "Verifica della pertinenza dei documenti...")
        documents, rejected = filter_documents(
            documents, author, request.get("era", ""), self.relevance_threshold
        )

        if rejected:
            self.logger.info(
                f"{len(rejected)} documenti scartati per scarsa pertinenza: "
                + ", ".join(title for title, _ in rejected[:5])
            )

        if not documents:
            self.error_occurred.emit(
                f"Trovati {len(rejected)} documenti ma nessuno pertinente a '{author}'. "
                "Provare a specificare meglio nome ed epoca, o abbassare "
                "research.relevance_threshold."
            )
            return []

        self.progress_update.emit(90, f"Salvataggio di {len(documents)} documenti...")
        paths = self._save(documents, author, rejected)

        primary = sum(1 for d in documents if d.is_primary_source)
        self.progress_update.emit(
            100,
            f"Ricerca completata: {len(documents)} documenti "
            f"({primary} fonti primarie, {len(documents) - primary} contestuali)",
        )
        self.stage_completed.emit(paths)
        return paths

    # ------------------------------------------------------------- supporto

    @staticmethod
    def _build_queries(author: str, era: str, extra: List[str] | None) -> List[str]:
        """Interrogazioni, tenute al minimo indispensabile.

        Il nome da solo e' gia' la ricerca piu' efficace; le varianti
        aggiungono poco e moltiplicano le chiamate a un servizio gratuito.
        L'epoca entra come secondo tentativo solo per disambiguare gli omonimi.
        """
        queries = [author]
        if era:
            queries.append(f"{author} {era}")
        if extra:
            queries.extend(str(q) for q in extra)

        seen: set[str] = set()
        return [q for q in queries if not (q in seen or seen.add(q))]

    @staticmethod
    def _default_languages(era: str) -> List[str]:
        """Lingue da interrogare, dedotte dall'epoca quando possibile.

        Per un personaggio dell'antichita' classica la fonte primaria e' in
        latino o greco: cercarla solo in italiano la mancherebbe.
        """
        era_lower = era.lower()
        languages = ["it", "en"]

        classical = ("roman", "romana", "latin", "antica roma", "repubblica", "impero",
                     "a.c.", "bc", "caesar", "cicero")
        greek = ("greek", "greca", "grecia", "hellenic", "ellenistic", "atene", "athens")

        if any(marker in era_lower for marker in classical):
            languages.append("la")
        if any(marker in era_lower for marker in greek):
            languages.append("el")

        return languages

    def _save(
        self,
        documents: List[ResearchDocument],
        author: str,
        rejected: List[tuple] | None = None,
    ) -> List[Path]:
        output_dir = self.project_dir / "sources" / "online"
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        manifest: List[Dict[str, Any]] = []

        for index, document in enumerate(documents, start=1):
            stem = _UNSAFE_FILENAME.sub("_", document.title)[:60].strip() or f"doc_{index}"
            path = output_dir / f"{index:03d}_{document.source}_{stem}.txt"

            try:
                path.write_text(document.text, encoding="utf-8")
            except OSError as exc:
                self.logger.warning(f"Salvataggio fallito per '{document.title}': {exc}")
                continue

            paths.append(path)
            manifest.append({
                "file": path.name,
                "title": document.title,
                "url": document.url,
                "source": document.source,
                "language": document.language,
                "is_primary_source": document.is_primary_source,
                "word_count": document.word_count,
                "relevance_score": document.metadata.get("relevance_score"),
            })

        # Il manifest tiene la provenienza di ogni riga del corpus: senza,
        # dopo l'ingestione non si saprebbe piu' da dove viene un testo.
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "author": author,
                    "retrieved_at": datetime.now().isoformat(),
                    "documents": manifest,
                    # Anche gli scarti restano tracciati: se un profilo esce
                    # povero, e' qui che si vede se il filtro e' stato severo.
                    "rejected": [
                        {"title": title, "score": verdict.score, "reasons": verdict.reasons}
                        for title, verdict in (rejected or [])
                    ],
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return paths
