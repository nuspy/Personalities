from typing import Dict, Any
from pathlib import Path
import logging
import sys

from ..stage_base import PipelineStage

logger = logging.getLogger(__name__)

class TrainingStage(PipelineStage):
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = project_dir
        
    def run(self, dataset_path: Path):
        """
        Runs the training process.
        Note: This blocks if run directly. Should be run in a QThread via the Orchestrator.
        """
        self.progress_update.emit(0, "Initializing Unsloth training...")
        
        try:
            # Import Unsloth locally to avoid crashes if not installed/supported
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import load_dataset
            import torch
        except ImportError as e:
            self.error_occurred.emit(f"Unsloth library not found or CUDA missing: {e}. Please install via 'pip install unsloth[colab-new]'")
            raise

        train_config = self.config.get("training", {})
        
        # 1. Load Model
        self.progress_update.emit(10, "Loading base model (Llama-3)...")
        
        max_seq_length = train_config.get("max_seq_length", 2048)
        dtype = None # Auto detection
        load_in_4bit = train_config.get("load_in_4bit", True)
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=train_config.get("base_model", "unsloth/Meta-Llama-3.1-8B-Instruct"),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # 2. Add LoRA Adapters
        self.progress_update.emit(20, "Applying LoRA adapters...")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=train_config['lora'].get('r', 16),
            target_modules=train_config['lora'].get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_alpha=train_config['lora'].get('alpha', 16),
            lora_dropout=train_config['lora'].get('dropout', 0),
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # 3. Load Dataset
        self.progress_update.emit(30, "Loading dataset...")
        try:
            # Expecting JSONL in ShareGPT format from Stage 3
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
            
            # Simple formatting function for Unsloth/Chat templates
            # Assuming ShareGPT format {"conversations": [...]}
            from unsloth.chat_templates import get_chat_template
            
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="llama-3",
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}, 
            )

            def formatting_prompts_func(examples):
                convs = examples["conversations"]
                texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in convs]
                return {"text": texts}
                
            dataset = dataset.map(formatting_prompts_func, batched=True)
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading dataset: {e}")
            raise

        # 4. Configure Trainer
        self.progress_update.emit(40, "Configuring trainer...")
        
        output_dir = self.project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=train_config.get("per_device_batch_size", 2),
                gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
                warmup_steps=5,
                max_steps=60, # Demo steps - normally use num_train_epochs
                learning_rate=train_config.get("learning_rate", 2e-4),
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=str(output_dir),
            ),
        )

        # 5. Train
        self.progress_update.emit(50, "Training started (this may take a while)...")
        trainer_stats = trainer.train()
        
        # 6. Save
        self.progress_update.emit(90, "Saving adapters...")
        model.save_pretrained(str(output_dir / "lora_adapters"))
        tokenizer.save_pretrained(str(output_dir / "lora_adapters"))
        
        # Merge option (not executed by default to save disk space)
        # model.push_to_hub_merged(...) or save_pretrained_merged(...)
        
        self.progress_update.emit(100, "Training complete!")
        self.stage_completed.emit({"output_path": str(output_dir / "lora_adapters"), "stats": trainer_stats})
