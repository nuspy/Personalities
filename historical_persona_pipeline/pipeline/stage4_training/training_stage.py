"""Stage 4 — addestramento, in quattro modalita'.

`LORA_NEW` addestra un adapter nuovo, `LORA_CONTINUE` ne riprende uno
esistente, `LORA_STACK` ne fonde uno o piu' nel modello base e addestra
sopra, `FULL_FINETUNE` tocca tutti i pesi.

Correzioni ereditate dalla versione precedente: `max_steps=60` era cablato e
rendeva `num_epochs` decorativo; `train_config['lora']` sollevava KeyError su
una configurazione parziale; `warmup_ratio`, `save_steps`, `save_total_limit`
e `weight_decay` erano dichiarati in configurazione e mai letti; il dataset
di validazione prodotto dallo stadio 3 non veniva mai usato.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..stage_base import PipelineStage
from .adapter_manager import (
    AdapterInfo, check_compatibility, load_and_merge_adapters, read_adapter,
)
from .training_modes import TrainingMode, TrainingPlan

logger = logging.getLogger(__name__)

# Corrispondenza fra i ruoli ShareGPT (usati nel dataset) e quelli standard
# attesi dai template di `transformers`.
SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bot": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def _to_standard_roles(conversation) -> list:
    """Converte i turni ShareGPT (`from`/`value`) in `role`/`content`.

    I turni gia' nel formato standard passano invariati: il dataset puo'
    arrivare da una versione precedente della pipeline o da una fonte esterna.
    """
    turns = []
    for turn in conversation:
        if "role" in turn and "content" in turn:
            turns.append({"role": turn["role"], "content": turn["content"]})
            continue
        role = SHAREGPT_ROLE_MAP.get(str(turn.get("from", "")).lower(), "user")
        turns.append({"role": role, "content": turn.get("value", "")})
    return turns


# Template di chat per famiglia di modelli, dedotto dal nome quando la
# configurazione dice "auto". Un template sbagliato non fa fallire il
# training: produce un modello che risponde con i token di controllo in chiaro.
CHAT_TEMPLATE_HINTS = (
    ("llama-3", ("llama-3", "llama3", "llama_3")),
    ("qwen-2.5", ("qwen2.5", "qwen-2.5", "qwen25")),
    ("qwen-2", ("qwen2", "qwen-2")),
    ("mistral", ("mistral", "mixtral")),
    ("gemma", ("gemma",)),
    ("phi-3", ("phi-3", "phi3")),
    ("chatml", ("chatml", "yi-", "openhermes")),
)


class TrainingStage(PipelineStage):
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = Path(project_dir)

    # ------------------------------------------------------------------ run

    def run(self, dataset_path: Path, plan: Optional[TrainingPlan] = None):
        """Esegue l'addestramento secondo il piano indicato.

        `plan` assente significa: leggilo dalla configurazione.
        """
        dataset_path = Path(dataset_path)
        plan = plan or TrainingPlan.from_config(self.config)

        problems = plan.validate()
        if problems:
            raise ValueError("Piano di addestramento non valido: " + " ".join(problems))

        self.progress_update.emit(0, f"Modalita': {plan.mode.label}")

        # Unsloth applica le sue ottimizzazioni con una patch a transformers,
        # trl e peft: se viene importato dopo di loro le patch non arrivano,
        # e lo segnala con un avviso invitando a spostare l'import in cima.
        # L'import qui e' comunque tardivo rispetto al modulo, ma precede
        # quello delle librerie che deve modificare.
        try:
            import unsloth  # noqa: F401
        except ImportError:
            pass  # si prosegue con transformers + peft

        try:
            import torch
            from datasets import load_dataset
            from trl import SFTTrainer
        except ImportError as exc:
            raise ImportError(
                f"Dipendenze di addestramento mancanti ({exc}). "
                "Installare con: pip install -e \".[training]\""
            ) from exc

        train_config = self.config.get("training", {})
        train_config = {**train_config, **plan.overrides}

        max_seq_length = int(train_config.get("max_seq_length", 2048))

        # 1. Modello base -----------------------------------------------------
        self.progress_update.emit(5, f"Carico il modello base: {plan.base_model}")
        model, tokenizer, backend = self._load_base_model(plan, train_config, max_seq_length)

        # 2. Adapter di partenza ---------------------------------------------
        active_adapter = ""
        if plan.adapters:
            self._verify_adapters(plan)

            if plan.mode is TrainingMode.LORA_STACK:
                # Gli adapter vengono fusi *nei pesi del modello* e poi
                # scaricati: quello nuovo parte da un modello gia' specializzato
                # e gli adapter di origine restano intatti su disco.
                self.progress_update.emit(15, "Fondo gli adapter di partenza nel modello...")
                model = self._merge_into_base(model, plan)
            else:
                self.progress_update.emit(15, "Carico l'adapter da proseguire...")
                model, active_adapter = load_and_merge_adapters(
                    model, plan.adapters, plan.merge_strategy,
                    progress=lambda msg: self.progress_update.emit(18, msg),
                )

        # 3. Adapter da addestrare -------------------------------------------
        if plan.mode in (TrainingMode.LORA_NEW, TrainingMode.LORA_STACK):
            self.progress_update.emit(22, "Applico un nuovo adapter LoRA...")
            model = self._attach_new_adapter(model, train_config, backend)
        elif plan.mode is TrainingMode.LORA_CONTINUE:
            self.progress_update.emit(22, f"Proseguo l'adapter '{active_adapter}'")
            self._ensure_trainable(model, backend)
        else:
            self.progress_update.emit(22, "Fine-tuning completo: tutti i pesi sono addestrabili")

        # 4. Dataset ----------------------------------------------------------
        self.progress_update.emit(30, "Preparo il dataset...")
        train_dataset, eval_dataset, tokenizer = self._prepare_datasets(
            dataset_path, tokenizer, plan, train_config, load_dataset, backend
        )

        # 5. Trainer ----------------------------------------------------------
        self.progress_update.emit(40, "Configuro il trainer...")
        output_dir = self.project_dir / "output" / plan.output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = self._build_trainer(
            SFTTrainer, model, tokenizer, train_dataset, eval_dataset,
            train_config, output_dir, torch, plan, max_seq_length,
        )

        # 6. Addestramento ----------------------------------------------------
        self.progress_update.emit(
            50, f"Addestramento avviato ({plan.mode.label}). Puo' richiedere molto tempo."
        )
        stats = trainer.train()

        # 7. Salvataggio ------------------------------------------------------
        self.progress_update.emit(92, "Salvo il risultato...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metadata = self._write_metadata(output_dir, plan, train_config, stats, dataset_path)

        self.progress_update.emit(100, f"Addestramento completato: {output_dir}")
        result = {
            "output_path": str(output_dir),
            "mode": plan.mode.value,
            "base_model": plan.base_model,
            "metadata": metadata,
            "stats": getattr(stats, "metrics", {}) or {},
        }
        self.stage_completed.emit(result)
        return result

    # --------------------------------------------------------------- modello

    def _load_base_model(
        self, plan: TrainingPlan, train_config: Dict[str, Any], max_seq_length: int
    ) -> Tuple[Any, Any, str]:
        """Carica il modello base con Unsloth se disponibile, altrimenti PEFT.

        Unsloth e' piu' veloce e usa meno memoria, ma non e' installabile
        ovunque (richiede CUDA e versioni precise). Il ripiego su
        transformers puro permette di addestrare anche dove Unsloth non c'e'.
        """
        # I downloader nativi di HuggingFace non usano i certificati di
        # sistema: su una rete con proxy TLS vanno disattivati prima del
        # primo scaricamento, non dopo che e' fallito.
        from ..utils.tls import prepare_model_download

        prepare_model_download()

        load_in_4bit = bool(train_config.get("load_in_4bit", True))
        full_finetune = plan.mode is TrainingMode.FULL_FINETUNE

        if full_finetune and load_in_4bit:
            # I pesi quantizzati a 4 bit non sono addestrabili direttamente:
            # con QLoRA si addestra l'adapter, non il modello.
            self.logger.warning(
                "Fine-tuning completo incompatibile con load_in_4bit: "
                "carico il modello in 16 bit."
            )
            load_in_4bit = False

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            self.logger.info("Unsloth non installato: uso transformers + peft.")
        else:
            kwargs: Dict[str, Any] = {
                "model_name": plan.base_model,
                "max_seq_length": max_seq_length,
                "dtype": None,  # rilevamento automatico
                "load_in_4bit": load_in_4bit,
            }
            try:
                # `full_finetuning` non esiste nelle versioni piu' vecchie.
                return (
                    *FastLanguageModel.from_pretrained(
                        **kwargs, full_finetuning=full_finetune
                    ),
                    "unsloth",
                )
            except TypeError:
                try:
                    return (*FastLanguageModel.from_pretrained(**kwargs), "unsloth")
                except Exception as exc:
                    self._report_unsloth_fallback(exc)
            except Exception as exc:
                # Unsloth ricompila i moduli del modello e la compilazione
                # puo' fallire per combinazioni di versioni (ad esempio
                # «BlockDiagonalCausalMask is not defined» quando xformers non
                # corrisponde). Non e' un motivo per fermare l'addestramento:
                # transformers carica lo stesso modello, solo piu' lentamente.
                self._report_unsloth_fallback(exc)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs: Dict[str, Any] = {"device_map": "auto", **_dtype_kwarg(dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(plan.base_model, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(plan.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, "transformers"

    def _report_unsloth_fallback(self, exc: Exception) -> None:
        """Spiega perche' si prosegue senza Unsloth invece di fermarsi."""
        message = (
            f"Unsloth non ha potuto caricare il modello ({type(exc).__name__}: "
            f"{str(exc)[:160]}). Proseguo con transformers + peft: "
            "l'addestramento e' piu' lento ma il risultato e' equivalente."
        )
        self.logger.warning(message)
        self.error_occurred.emit(f"Avviso: {message}")

    def _verify_adapters(self, plan: TrainingPlan) -> None:
        """Blocca prima di iniziare se gli adapter non sono compatibili."""
        infos = []
        for source in plan.adapters:
            info = read_adapter(Path(source.path))
            if info is None:
                raise FileNotFoundError(
                    f"'{source.path}' non contiene un adapter LoRA "
                    f"(manca adapter_config.json)."
                )
            infos.append(info)

        errors, warnings = check_compatibility(infos, plan.base_model)
        for warning in warnings:
            self.logger.warning(warning)
            self.error_occurred.emit(f"Avviso: {warning}")
        if errors:
            raise ValueError(" ".join(errors))

        for info in infos:
            self.logger.info(f"Adapter di partenza: {info.name} ({info.describe()})")

    def _merge_into_base(self, model, plan: TrainingPlan):
        """Fonde gli adapter nei pesi del modello e li scarica.

        Dopo questa operazione il modello *contiene* cio' che gli adapter
        avevano appreso, ma non ha piu' adapter attivi: il successivo puo'
        essere applicato pulito.
        """
        model, _ = load_and_merge_adapters(
            model, plan.adapters, plan.merge_strategy,
            progress=lambda msg: self.progress_update.emit(17, msg),
        )
        merged = model.merge_and_unload()
        self.logger.info("Adapter fusi nei pesi del modello base")
        return merged

    def _attach_new_adapter(self, model, train_config: Dict[str, Any], backend: str):
        lora = train_config.get("lora", {})
        target_modules = lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        rank = int(lora.get("r", 16))
        alpha = int(lora.get("alpha", rank * 2))
        dropout = float(lora.get("dropout", 0.0))
        seed = int(self.config.get("advanced", {}).get("seed", 3407))

        if backend == "unsloth":
            from unsloth import FastLanguageModel

            return FastLanguageModel.get_peft_model(
                model,
                r=rank,
                target_modules=target_modules,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=seed,
                use_rslora=bool(lora.get("use_rslora", False)),
                loftq_config=None,
            )

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model)

        return get_peft_model(model, LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=bool(lora.get("use_rslora", False)),
        ))

    def _ensure_trainable(self, model, backend: str) -> None:
        """In `LORA_CONTINUE` l'adapter caricato deve tornare addestrabile.

        Due condizioni, entrambe necessarie:

        1. i parametri dell'adapter devono avere `requires_grad`.
           `PeftModel.from_pretrained` li carica congelati salvo
           `is_trainable`, e senza questo il training girerebbe senza
           aggiornare nulla, terminando con successo e lasciando l'adapter
           identico a prima;

        2. gli **input embedding** devono propagare il gradiente. Col
           gradient checkpointing gli attivazioni intermedie non vengono
           conservate ma ricalcolate, e se l'ingresso del blocco ricalcolato
           non richiede gradiente il grafo si interrompe: l'errore che ne
           esce, «element 0 of tensors does not require grad», non nomina ne'
           gli adapter ne' il checkpointing e non aiuta a capire.
        """
        if backend == "unsloth":
            # Unsloth tiene il modello in modalita' inferenza dopo il
            # caricamento: questa chiamata lo riporta in addestramento e
            # sistema da sola i requires_grad.
            try:
                from unsloth import FastLanguageModel

                FastLanguageModel.for_training(model)
            except Exception as exc:
                self.logger.debug(f"for_training non applicabile: {exc}")

        if not any(p.requires_grad for p in model.parameters()):
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(
                "Nessun parametro addestrabile dopo il caricamento dell'adapter: "
                "l'addestramento non produrrebbe alcun cambiamento."
            )

        # Necessario col gradient checkpointing; innocuo altrimenti.
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception as exc:
                self.logger.debug(f"enable_input_require_grads non riuscita: {exc}")

    # --------------------------------------------------------------- dataset

    def _prepare_datasets(
        self, dataset_path: Path, tokenizer, plan: TrainingPlan,
        train_config: Dict[str, Any], load_dataset, backend: str,
    ):
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

        tokenizer = self._apply_chat_template(tokenizer, plan, train_config, backend)

        train_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        # Il file di validazione prodotto dallo stadio 3 non veniva mai usato.
        eval_dataset = None
        val_path = dataset_path.parent / "val_dataset.jsonl"
        if val_path.exists():
            eval_dataset = load_dataset("json", data_files=str(val_path), split="train")
            self.logger.info(f"Dataset di validazione: {len(eval_dataset)} esempi")

        # I template di chat leggono `role`/`content`. Il dataset e' in formato
        # ShareGPT (`from`/`value`): passarglielo cosi' com'e' non solleva
        # alcun errore — il template non trova i campi che cerca, scarta la
        # conversazione e restituisce il prompt di sistema di default del
        # modello. Il risultato sono N esempi identici e privi del personaggio,
        # con l'addestramento che termina regolarmente e un adapter vuoto.
        #
        # La conversione si applica SEMPRE, anche con Unsloth. Il parametro
        # `mapping` di `get_chat_template` sembra fatto apposta per questo, ma
        # verificandolo su Unsloth 2026.9 si vede che non converte i dati:
        # riceve i turni ShareGPT e ne tiene solo quelli il cui ruolo coincide
        # gia' con lo standard (`system`), scartando `human` e `gpt`. Affidarsi
        # a quel parametro riproduce esattamente il difetto descritto sopra.
        def format_conversations(examples):
            texts = []
            for conversation in examples["conversations"]:
                texts.append(tokenizer.apply_chat_template(
                    _to_standard_roles(conversation),
                    tokenize=False,
                    add_generation_prompt=False,
                ))
            return {"text": texts}

        train_dataset = train_dataset.map(format_conversations, batched=True)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(format_conversations, batched=True)

        self._verify_formatting(train_dataset)

        self.logger.info(f"Dataset di addestramento: {len(train_dataset)} esempi")
        return train_dataset, eval_dataset, tokenizer

    def _verify_formatting(self, dataset) -> None:
        """Controlla che il testo formattato contenga davvero le conversazioni.

        Un template applicato al formato sbagliato produce testo plausibile ma
        vuoto di contenuto, e l'errore si manifesta solo molte ore dopo, come
        un modello che non ha imparato niente. Qui si confronta un campione:
        il testo deve contenere la risposta dell'assistente da cui deriva.
        """
        if len(dataset) == 0:
            raise ValueError("Dataset vuoto dopo la formattazione")

        sample = dataset[0]
        text = sample.get("text", "")
        if not text.strip():
            raise ValueError("La formattazione ha prodotto testo vuoto")

        expected = ""
        for turn in sample.get("conversations", []):
            value = turn.get("value") or turn.get("content") or ""
            if turn.get("from") in ("gpt", "assistant") or turn.get("role") == "assistant":
                expected = value
                break

        if expected:
            # Si confronta un frammento: il template puo' normalizzare spazi
            # e aggiungere token di controllo attorno al contenuto.
            probe = expected.strip()[:40]
            if probe and probe not in text:
                raise ValueError(
                    "Il testo formattato non contiene le conversazioni del dataset: "
                    "il template di chat non riconosce il formato dei turni. "
                    f"Atteso un frammento come «{probe}», ottenuto: «{text[:160]}»"
                )

        # Testi tutti uguali: sintomo tipico di un template che ignora i turni.
        if len(dataset) > 1:
            distinct = len({dataset[i]["text"] for i in range(min(8, len(dataset)))})
            if distinct == 1:
                self.logger.warning(
                    "Tutti gli esempi formattati sono identici: verificare il "
                    "dataset e il template di chat."
                )

    def _apply_chat_template(self, tokenizer, plan: TrainingPlan,
                             train_config: Dict[str, Any], backend: str):
        """Imposta il template di chat, dedotto dal modello se non specificato."""
        template = str(train_config.get("chat_template", "auto"))
        if template == "auto":
            template = self._guess_chat_template(plan.base_model)
            self.logger.info(f"Template di chat dedotto dal modello: {template}")

        if backend != "unsloth":
            # Con transformers puro si usa il template gia' presente nel
            # tokenizer, che per i modelli instruct e' quello corretto.
            if getattr(tokenizer, "chat_template", None):
                return tokenizer
            self.logger.warning(
                "Il tokenizer non ha un template di chat e Unsloth non e' "
                "disponibile per fornirne uno: il formato potrebbe essere errato."
            )
            return tokenizer

        from unsloth.chat_templates import get_chat_template

        # Nessun `mapping`: i turni arrivano gia' convertiti in `role`/`content`
        # da `_to_standard_roles`. Passarlo darebbe l'impressione che la
        # conversione avvenga qui, mentre non avviene (vedi `_prepare_datasets`).
        return get_chat_template(tokenizer, chat_template=template)

    @staticmethod
    def _guess_chat_template(model_name: str) -> str:
        lowered = model_name.lower()
        for template, markers in CHAT_TEMPLATE_HINTS:
            if any(marker in lowered for marker in markers):
                return template
        return "chatml"

    # --------------------------------------------------------------- trainer

    def _build_trainer(
        self, SFTTrainer, model, tokenizer, train_dataset, eval_dataset,
        train_config: Dict[str, Any], output_dir: Path, torch,
        plan: TrainingPlan, max_seq_length: int,
    ):
        """Costruisce il trainer adattandosi alla versione di TRL installata.

        TRL ha spostato piu' volte gli stessi parametri: `dataset_text_field`,
        `max_seq_length` e `packing` stavano su `SFTTrainer`, poi sono passati
        a `SFTConfig`, dove `max_seq_length` e' inoltre diventato `max_length`;
        `tokenizer` e' diventato `processing_class`. Passare l'argomento
        sbagliato solleva un TypeError dopo che il modello e' gia' in memoria.

        Qui le firme si leggono a runtime, cosi' il codice regge sia le
        versioni gia' installate sia quelle future entro il range dichiarato.
        """
        import inspect

        args = self._build_training_arguments(
            train_config, output_dir, torch, bool(eval_dataset), plan
        )

        # Parametri che, a seconda della versione, vivono su SFTConfig o sul
        # costruttore del trainer.
        dataset_params = {
            "dataset_text_field": "text",
            "packing": False,
            "dataset_num_proc": 2,
        }
        length_params = {"max_seq_length": max_seq_length, "max_length": max_seq_length}

        config_fields = set(getattr(type(args), "__dataclass_fields__", {}))
        for name, value in {**dataset_params, **length_params}.items():
            if name in config_fields:
                setattr(args, name, value)

        trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
        kwargs: Dict[str, Any] = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
        }
        if eval_dataset is not None:
            kwargs["eval_dataset"] = eval_dataset

        # `tokenizer` e' stato rinominato `processing_class`.
        if "processing_class" in trainer_params:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            kwargs["tokenizer"] = tokenizer

        for name, value in {**dataset_params, **length_params}.items():
            if name in trainer_params and name not in config_fields:
                kwargs[name] = value

        self.logger.debug(f"SFTTrainer con: {sorted(kwargs)}")
        return SFTTrainer(**kwargs)

    def _build_training_arguments(
        self, train_config: Dict[str, Any], output_dir: Path,
        torch, has_eval: bool, plan: TrainingPlan,
    ):
        """Configurazione del trainer: `SFTConfig` se TRL lo espone.

        `SFTConfig` estende `TrainingArguments`, quindi accetta gli stessi
        campi piu' quelli specifici dell'SFT: usarlo direttamente evita di
        doverli impostare altrove.
        """
        try:
            from trl import SFTConfig as ArgumentsClass
        except ImportError:
            from transformers import TrainingArguments as ArgumentsClass

        advanced = self.config.get("advanced", {})
        max_steps = int(train_config.get("max_steps", -1))

        # Il fine-tuning completo aggiorna tutti i pesi: con il learning rate
        # di una LoRA (2e-4) il modello si degrada. Si scende di un ordine di
        # grandezza, come e' prassi.
        default_lr = 2e-5 if plan.mode is TrainingMode.FULL_FINETUNE else 2e-4
        learning_rate = float(train_config.get("learning_rate", default_lr))
        if plan.mode is TrainingMode.FULL_FINETUNE and learning_rate > 5e-5:
            self.logger.warning(
                f"learning_rate={learning_rate} e' alto per un fine-tuning completo: "
                f"riduco a {default_lr}. Impostare 'training.learning_rate' per forzarlo."
            )
            learning_rate = default_lr

        args: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": int(train_config.get("per_device_batch_size", 2)),
            "gradient_accumulation_steps": int(train_config.get("gradient_accumulation_steps", 4)),
            "warmup_ratio": float(train_config.get("warmup_ratio", 0.03)),
            "learning_rate": learning_rate,
            "weight_decay": float(train_config.get("weight_decay", 0.01)),
            "lr_scheduler_type": str(train_config.get("lr_scheduler", "linear")),
            "logging_steps": int(advanced.get("logging_steps", 10)),
            "save_steps": int(train_config.get("save_steps", 100)),
            "save_total_limit": int(train_config.get("save_total_limit", 3)),
            "seed": int(advanced.get("seed", 3407)),
            "optim": str(train_config.get("optimizer", "adamw_8bit")),
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "gradient_checkpointing": bool(train_config.get("gradient_checkpointing", True)),
            "report_to": "none",
        }

        if max_steps > 0:
            args["max_steps"] = max_steps
            self.logger.warning(
                f"max_steps={max_steps}: num_epochs viene ignorato. "
                "Usare -1 per addestrare per epoche."
            )
        else:
            args["num_train_epochs"] = float(train_config.get("num_epochs", 3))

        if has_eval:
            args["eval_strategy"] = "steps"
            args["eval_steps"] = int(advanced.get("eval_steps", 100))

        # `eval_strategy` si chiamava `evaluation_strategy` prima di
        # transformers 4.46, e passare il nome sbagliato e' un errore fatale.
        supported = set(getattr(ArgumentsClass, "__dataclass_fields__", {}))
        if supported:
            if "eval_strategy" in args and "eval_strategy" not in supported:
                args["evaluation_strategy"] = args.pop("eval_strategy")
            unsupported = [k for k in args if k not in supported]
            for key in unsupported:
                self.logger.debug(f"Parametro non supportato da questa versione, ignorato: {key}")
                args.pop(key)

        return ArgumentsClass(**args)

    # -------------------------------------------------------------- metadati

    def _write_metadata(
        self, output_dir: Path, plan: TrainingPlan,
        train_config: Dict[str, Any], stats, dataset_path: Path,
    ) -> Dict[str, Any]:
        """Registra come e' stato prodotto questo adapter.

        Senza questo file, un adapter in una cartella e' indistinguibile da
        un altro: non si sa su quale base sia stato addestrato ne' da quali
        adapter derivi, e la catena delle fusioni diventa irricostruibile.
        """
        metadata = {
            "created_at": datetime.now().isoformat(),
            "mode": plan.mode.value,
            "base_model": plan.base_model,
            "author_name": self.config.get("persona", {}).get("author_name", ""),
            "dataset": str(dataset_path),
            "derived_from": [
                {"name": a.name, "path": str(a.path), "weight": a.weight}
                for a in plan.adapters
            ],
            "merge_strategy": plan.merge_strategy if len(plan.adapters) > 1 else None,
            "lora": train_config.get("lora", {}) if plan.mode.produces_adapter else None,
            "epochs": train_config.get("num_epochs"),
            "learning_rate": train_config.get("learning_rate"),
            "final_loss": (getattr(stats, "metrics", {}) or {}).get("train_loss"),
        }

        (output_dir / "persona_training.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return metadata


def _dtype_kwarg(dtype) -> Dict[str, Any]:
    """Nome del parametro del tipo numerico, secondo la versione di transformers.

    `torch_dtype` e' stato rinominato `dtype` in transformers 4.56: passare il
    vecchio nome funziona ancora ma stampa un avviso di deprecazione a ogni
    caricamento, e smettera' di funzionare.
    """
    import transformers
    from transformers import AutoModelForCausalLM

    fields = getattr(AutoModelForCausalLM.from_pretrained, "__doc__", "") or ""
    version = tuple(int(p) for p in transformers.__version__.split(".")[:2] if p.isdigit())
    if version >= (4, 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}
