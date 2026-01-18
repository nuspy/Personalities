from typing import Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime
import json
import uuid

from ..stage_base import PipelineStage
from ..data_models import CompleteStyleProfile, TrainingDataset, IngestionResult
from ..utils.llm_provider import LLMFactory
from .conversation_generator import ConversationGenerator

logger = logging.getLogger(__name__)

class DatasetStage(PipelineStage):
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = project_dir
        
    def run(self, input_data: Dict[str, Any]) -> TrainingDataset:
        """
        Expects input_data to contain both 'profile' and 'ingestion_result'
        """
        profile = input_data.get('profile')
        ingestion_result = input_data.get('ingestion_result')
        
        if not profile or not isinstance(profile, CompleteStyleProfile):
             raise ValueError("Missing or invalid Style Profile")
        
        self.progress_update.emit(0, "Initializing dataset generation...")
        
        # 1. Setup LLM Provider
        try:
            llm_provider = LLMFactory.create(self.config.get("dataset", {}))
            generator = ConversationGenerator(llm_provider, self.config)
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize LLM Provider: {e}")
            raise

        total_target = self.config["dataset"].get("num_conversations", 100)
        all_conversations = []
        
        # 2. Content-Based Generation (from Ingestion Segments) - 50% of target
        if ingestion_result and ingestion_result.segments:
            content_count = int(total_target * 0.5)
            self.progress_update.emit(10, f"Generating {content_count} content-based examples from source text...")
            
            content_convs = generator.generate_from_segments(
                profile, 
                ingestion_result.segments, 
                content_count
            )
            all_conversations.extend(content_convs)
            logger.info(f"Generated {len(content_convs)} content-based conversations")
        
        # 3. Synthetic Generation (Values, Anachronisms, etc.) - Remaining 50%
        types_dist = {
            "philosophy_values": 0.3,
            "anachronistic": 0.2
        }
        
        remaining_target = total_target - len(all_conversations)
        
        for idx, (category, ratio) in enumerate(types_dist.items()):
            count = int(total_target * ratio)
            # Adjust if we need to fill up to target
            if idx == len(types_dist) - 1:
                count = max(count, total_target - len(all_conversations))
            
            if count <= 0: continue
                
            self.progress_update.emit(
                50 + int((idx / len(types_dist)) * 40), 
                f"Generating {count} synthetic examples for {category}..."
            )
            
            # Generate in chunks
            chunk_size = 5
            generated = 0
            while generated < count:
                batch = min(chunk_size, count - generated)
                new_convs = generator.generate_batch(profile, category, batch)
                if not new_convs: break
                all_conversations.extend(new_convs)
                generated += len(new_convs)
        
        # 4. Save and Finalize
        self.progress_update.emit(95, "Formatting dataset...")
        
        dataset = TrainingDataset(
            project_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            author_name=profile.author_name,
            total_conversations=len(all_conversations),
            conversations=all_conversations,
            type_distribution={"mixed": len(all_conversations)}
        )
        
        self._save_dataset(dataset)
        
        self.progress_update.emit(100, "Dataset generation complete")
        self.stage_completed.emit(dataset)
        return dataset
    
    def _save_dataset(self, dataset: TrainingDataset):
        output_dir = self.project_dir / "datasets"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "raw_dataset.json", 'w', encoding='utf-8') as f:
            f.write(dataset.model_dump_json(indent=2))
            
        jsonl_path = output_dir / "train_dataset.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for conv in dataset.conversations:
                entry = {
                    "conversations": [
                        {"from": "system", "value": conv.system_prompt},
                    ]
                }
                for turn in conv.turns:
                    role_map = {"user": "human", "assistant": "gpt"}
                    entry["conversations"].append({
                        "from": role_map.get(turn["role"], "human"),
                        "value": turn["content"]
                    })
                f.write(json.dumps(entry) + "\n")