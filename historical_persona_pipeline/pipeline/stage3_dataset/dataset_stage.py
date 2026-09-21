"""Stage 3 — dataset: dalle domande alle conversazioni di addestramento.

Correzioni rispetto alla versione precedente:

- le `conversation_types` della configurazione vengono **usate**: prima
  erano sei in configurazione e due cablate nel codice;
- `val_split` produce davvero un file di validazione, prima era ignorato;
- `self.config["dataset"]` sollevava KeyError su una configurazione parziale;
- la raggiungibilita' dell'LLM si verifica **prima** di iniziare, non dopo
  venti minuti di tentativi a vuoto;
- l'avanzamento riflette le conversazioni prodotte, non i passi dello stadio.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..data_models import (
    CompleteStyleProfile,
    IngestionResult,
    TrainingConversation,
    TrainingDataset,
)
from ..stage_base import PipelineStage
from ..utils.llm_provider import LLMError, LLMFactory
from .conversation_generator import ConversationGenerator
from .question_generator import QuestionGenerator

logger = logging.getLogger(__name__)


class DatasetStage(PipelineStage):
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = Path(project_dir)

    def run(self, input_data: Dict[str, Any]) -> TrainingDataset:
        profile = input_data.get("profile")
        ingestion_result = input_data.get("ingestion_result")

        if not isinstance(profile, CompleteStyleProfile):
            raise ValueError("Profilo di stile mancante o non valido")

        dataset_config = self.config.get("dataset", {})
        target = int(dataset_config.get("num_conversations", 100))

        self.progress_update.emit(0, "Preparazione della generazione...")

        llm_provider = LLMFactory.create(dataset_config)

        # Un endpoint spento deve emergere subito: senza questo controllo
        # l'errore arriva dopo centinaia di tentativi falliti.
        reachable, detail = llm_provider.health_check()
        if not reachable:
            raise LLMError(
                f"Servizio LLM non raggiungibile ({detail}). "
                f"Verificare dataset.llm_base_url e che il modello sia caricato."
            )

        self.progress_update.emit(5, f"Generazione di {target} domande...")

        question_generator = QuestionGenerator(self.config, llm_provider)
        segments = ingestion_result.segments if isinstance(ingestion_result, IngestionResult) else []
        questions = question_generator.generate(
            profile, segments, target,
            distribution=dataset_config.get("conversation_types"),
        )

        if not questions:
            raise ValueError(
                "Nessuna domanda generata: il profilo e il corpus non offrono "
                "materiale sufficiente."
            )

        self.progress_update.emit(
            12, f"{len(questions)} domande pronte. Generazione delle risposte..."
        )

        generator = ConversationGenerator(llm_provider, self.config)

        started = time.monotonic()
        estimate_announced = False

        def on_progress(done: int, total: int) -> None:
            nonlocal estimate_announced

            # 12% -> 92%: la generazione e' la parte lunga dello stadio.
            percent = 12 + int(done / max(total, 1) * 80)
            message = f"Conversazioni generate: {done}/{total}"

            # Il ritmo dipende dal modello e dall'hardware, e puo' variare di
            # due ordini di grandezza: con un modello grande in locale mille
            # conversazioni sono ore. Dopo qualche esempio il ritmo e' noto,
            # e vale la pena dirlo subito — chi non se lo aspetta scopre la
            # durata solo restando a guardare.
            if done >= 3 and not estimate_announced:
                estimate_announced = True
                per_item = (time.monotonic() - started) / done
                remaining = per_item * (total - done)
                if remaining > 300:
                    self.error_occurred.emit(
                        f"Avviso: al ritmo attuale ({per_item:.0f}s per conversazione) "
                        f"restano circa {_human_duration(remaining)}. "
                        "Per ridurre i tempi: abbassare dataset.num_conversations, "
                        "alzare dataset.max_workers, o usare un modello piu' piccolo."
                    )

            if done and done % 10 == 0:
                per_item = (time.monotonic() - started) / done
                message += f" (~{per_item:.0f}s ciascuna)"

            self.progress_update.emit(percent, message)

        conversations = generator.generate(profile, questions, on_progress)

        if not conversations:
            # Senza i motivi, «nessuna conversazione prodotta» non dice nulla
            # di azionabile: il problema puo' essere il budget di token, il
            # formato della risposta, un rifiuto del modello o la rete.
            reasons = generator.failure_summary()
            raise ValueError(
                "Nessuna conversazione utilizzabile prodotta"
                + (f". Motivi ricorrenti: {reasons}" if reasons else ".")
            )

        discarded = sum(generator.last_failures.values())
        if discarded:
            self.error_occurred.emit(
                f"Avviso: {discarded} risposte su {len(questions)} scartate. "
                f"Motivi: {generator.failure_summary()}"
            )

        self.progress_update.emit(93, "Scrittura del dataset...")

        dataset = TrainingDataset(
            project_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            author_name=profile.author_name,
            total_conversations=len(conversations),
            conversations=conversations,
            type_distribution=self._type_distribution(conversations),
        )

        train_count, val_count = self._save(dataset, dataset_config)

        self.progress_update.emit(
            100,
            f"Dataset completato: {train_count} esempi di addestramento, "
            f"{val_count} di validazione",
        )
        self.stage_completed.emit(dataset)
        return dataset

    # -------------------------------------------------------------- supporto

    @staticmethod
    def _type_distribution(conversations: List[TrainingConversation]) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for conversation in conversations:
            distribution[conversation.conversation_type] = (
                distribution.get(conversation.conversation_type, 0) + 1
            )
        return distribution

    def _save(
        self, dataset: TrainingDataset, dataset_config: Dict[str, Any]
    ) -> Tuple[int, int]:
        output_dir = self.project_dir / "datasets"
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "raw_dataset.json").write_text(
            dataset.model_dump_json(indent=2), encoding="utf-8"
        )

        conversations = list(dataset.conversations)
        val_split = float(dataset_config.get("val_split", 0.1))

        # Divisione stratificata: se la validazione contenesse solo esempi
        # anacronistici la valutazione misurerebbe una cosa sola.
        train, validation = self._stratified_split(conversations, val_split)

        self._write_jsonl(output_dir / "train_dataset.jsonl", train)
        if validation:
            self._write_jsonl(output_dir / "val_dataset.jsonl", validation)

        self.logger.info(
            f"Dataset salvato in {output_dir}: {len(train)} train, {len(validation)} val"
        )
        return len(train), len(validation)

    @staticmethod
    def _stratified_split(
        conversations: List[TrainingConversation], val_split: float
    ) -> Tuple[List[TrainingConversation], List[TrainingConversation]]:
        if val_split <= 0 or len(conversations) < 10:
            return conversations, []

        by_type: Dict[str, List[TrainingConversation]] = {}
        for conversation in conversations:
            by_type.setdefault(conversation.conversation_type, []).append(conversation)

        train: List[TrainingConversation] = []
        validation: List[TrainingConversation] = []

        for group in by_type.values():
            cut = int(len(group) * val_split)
            validation.extend(group[:cut])
            train.extend(group[cut:])

        return train, validation

    @staticmethod
    def _write_jsonl(path: Path, conversations: List[TrainingConversation]) -> None:
        """Formato ShareGPT, atteso dai template di chat di Unsloth."""
        role_map = {"user": "human", "assistant": "gpt", "system": "system"}

        with open(path, "w", encoding="utf-8") as fh:
            for conversation in conversations:
                turns = [{"from": "system", "value": conversation.system_prompt}]
                for turn in conversation.turns:
                    turns.append({
                        "from": role_map.get(turn.get("role", "user"), "human"),
                        "value": turn.get("content", ""),
                    })
                fh.write(json.dumps({"conversations": turns}, ensure_ascii=False) + "\n")


def _human_duration(seconds: float) -> str:
    """Durata in forma leggibile: i secondi non dicono nulla oltre il minuto."""
    if seconds < 90:
        return f"{seconds:.0f} secondi"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minuti"
    return f"{minutes / 60:.1f} ore"
