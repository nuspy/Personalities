"""Conversione: da adapter a modello utilizzabile altrove.

Copre quattro operazioni che prima non esistevano:

1. **fusione** di un adapter nel modello base, con esito un modello autonomo;
2. **esportazione GGUF** con quantizzazione, per llama.cpp / LM Studio / Ollama;
3. **fusione di piu' adapter** in uno solo, senza addestrare;
4. **pacchetto Ollama**, col prompt di sistema del personaggio gia' dentro.

Due backend: Unsloth quando c'e' (piu' veloce, meno memoria), altrimenti
transformers + peft, che funzionano ovunque — anche su CPU, lentamente. La
conversione GGUF richiede in piu' gli strumenti di llama.cpp, e se mancano lo
si dice prima di iniziare invece di fallire a meta'.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..stage_base import PipelineStage
from ..stage4_training.adapter_manager import (
    check_compatibility, load_and_merge_adapters, read_adapter,
)
from ..stage4_training.training_modes import AdapterSource
from .formats import (
    DEFAULT_QUANTIZATION, ExportFormat, QUANTIZATIONS,
    estimate_parameters_billions,
)

logger = logging.getLogger(__name__)

# Nomi possibili dello script di conversione di llama.cpp: e' stato
# rinominato fra le versioni, e cercarne uno solo fallisce sulle altre.
GGUF_CONVERT_SCRIPTS = (
    "convert_hf_to_gguf.py",
    "convert-hf-to-gguf.py",
    "convert.py",
)

GGUF_QUANTIZE_BINARIES = ("llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe")


class ConversionStage(PipelineStage):
    """Converte adapter e modelli nei formati distribuibili."""

    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = Path(project_dir)
        self.conversion_config = config.get("conversion", {})

    # ------------------------------------------------------------------ run

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue una conversione.

        `request` accetta:
          - `adapters`: percorsi degli adapter (uno o piu')
          - `base_model`: modello base; dedotto dall'adapter se assente
          - `format`: valore di `ExportFormat`
          - `quantization`: livello GGUF
          - `output_dir`: destinazione
          - `merge_strategy`, `system_prompt`, `model_name`
        """
        export_format = ExportFormat(request.get("format", ExportFormat.MERGED_16BIT.value))
        adapter_paths = [Path(p) for p in request.get("adapters", []) if p]

        if not adapter_paths:
            raise ValueError("Nessun adapter indicato per la conversione.")

        sources = self._build_sources(adapter_paths, request.get("weights"))
        base_model = request.get("base_model") or self._infer_base_model(adapter_paths)
        if not base_model:
            raise ValueError(
                "Modello base non indicato e non deducibile dall'adapter "
                "(adapter_config.json privo di base_model_name_or_path)."
            )

        output_dir = Path(
            request.get("output_dir")
            or self.project_dir / "exports" / f"{export_format.value}_{_timestamp()}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        self.progress_update.emit(0, f"Conversione: {export_format.label}")
        self._check_disk_space(base_model, export_format, output_dir)

        if export_format is ExportFormat.ADAPTER_ONLY:
            result = self._export_adapter_only(sources, base_model, output_dir, request)
        elif export_format.needs_gguf_toolchain:
            result = self._export_gguf(sources, base_model, output_dir, request, export_format)
        else:
            result = self._export_merged(sources, base_model, output_dir, request, export_format)

        self.progress_update.emit(100, f"Conversione completata: {output_dir}")
        self.stage_completed.emit(result)
        return result

    # ------------------------------------------------------- solo adapter

    def _export_adapter_only(
        self, sources: List[AdapterSource], base_model: str,
        output_dir: Path, request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Copia l'adapter, fondendo prima i multipli in uno solo."""
        if len(sources) == 1:
            self.progress_update.emit(30, "Copio l'adapter...")
            source_dir = Path(sources[0].path)
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, output_dir / item.name)
        else:
            self.progress_update.emit(20, f"Fondo {len(sources)} adapter...")
            model, tokenizer = self._load_for_merge(base_model, load_in_4bit=True)
            model, _ = load_and_merge_adapters(
                model, sources, request.get("merge_strategy", "linear"),
                progress=lambda msg: self.progress_update.emit(40, msg),
            )
            self.progress_update.emit(70, "Salvo l'adapter fuso...")
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

        self._write_manifest(output_dir, sources, base_model, ExportFormat.ADAPTER_ONLY, request)
        return {
            "format": ExportFormat.ADAPTER_ONLY.value,
            "output_path": str(output_dir),
            "base_model": base_model,
        }

    # ------------------------------------------------------- modello fuso

    def _export_merged(
        self, sources: List[AdapterSource], base_model: str, output_dir: Path,
        request: Dict[str, Any], export_format: ExportFormat,
    ) -> Dict[str, Any]:
        four_bit = export_format is ExportFormat.MERGED_4BIT

        self.progress_update.emit(10, f"Carico il modello base: {base_model}")
        model, tokenizer = self._load_for_merge(base_model, load_in_4bit=four_bit)

        self.progress_update.emit(35, "Applico gli adapter...")
        model, _ = load_and_merge_adapters(
            model, sources, request.get("merge_strategy", "linear"),
            progress=lambda msg: self.progress_update.emit(45, msg),
        )

        self.progress_update.emit(60, "Fondo gli adapter nei pesi del modello...")

        # `merge_and_unload` di PEFT e' il percorso usato sempre: incorpora i
        # pesi dell'adapter e *rimuove* i moduli che lo avvolgono, lasciando un
        # modello con la struttura originale.
        #
        # Il metodo `save_pretrained_merged` di Unsloth sembrerebbe equivalente,
        # ma applicato a un modello avvolto da PEFT lascia nei pesi la struttura
        # dell'adapter (`...q_proj.base_layer.weight`). Il salvataggio riesce, e
        # il problema emerge molto dopo: la conversione GGUF si ferma con
        # «Can not map tensor 'model.layers.0.self_attn.q_proj.base_layer.bias'».
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
        elif hasattr(model, "save_pretrained_merged"):
            save_method = "merged_4bit_forced" if four_bit else "merged_16bit"
            model.save_pretrained_merged(str(output_dir), tokenizer, save_method=save_method)
            self._verify_merged(output_dir)
            self._write_manifest(output_dir, sources, base_model, export_format, request)
            return {
                "format": export_format.value,
                "output_path": str(output_dir),
                "base_model": base_model,
            }
        else:
            raise RuntimeError(
                "Il modello caricato non espone un metodo di fusione: "
                "verificare la versione di peft."
            )

        self.progress_update.emit(80, "Salvo il modello...")
        merged.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

        self._verify_merged(output_dir)
        self._write_manifest(output_dir, sources, base_model, export_format, request)
        return {
            "format": export_format.value,
            "output_path": str(output_dir),
            "base_model": base_model,
        }

    # -------------------------------------------------------------- GGUF

    def _export_gguf(
        self, sources: List[AdapterSource], base_model: str, output_dir: Path,
        request: Dict[str, Any], export_format: ExportFormat,
    ) -> Dict[str, Any]:
        quantization = str(request.get("quantization", DEFAULT_QUANTIZATION))
        if quantization not in QUANTIZATIONS:
            raise ValueError(
                f"Quantizzazione sconosciuta: '{quantization}'. "
                f"Valide: {', '.join(QUANTIZATIONS)}."
            )

        # Il modello va prima fuso: GGUF non conosce gli adapter.
        merged_dir = output_dir / "merged_fp16"
        self.progress_update.emit(5, "Fondo l'adapter prima della conversione...")
        # `_write_manifest` dell'intermedio e' superfluo — quello definitivo
        # viene scritto in fondo — ma resta come traccia se la conversione si
        # interrompe a meta' e la cartella intermedia sopravvive.
        self._export_merged(
            sources, base_model, merged_dir, request, ExportFormat.MERGED_16BIT
        )

        self.progress_update.emit(50, f"Converto in GGUF ({quantization})...")
        gguf_path = self._run_gguf_conversion(merged_dir, output_dir, quantization, request)

        result: Dict[str, Any] = {
            "format": export_format.value,
            "output_path": str(output_dir),
            "gguf_path": str(gguf_path),
            "quantization": quantization,
            "base_model": base_model,
        }

        if export_format is ExportFormat.OLLAMA:
            self.progress_update.emit(90, "Scrivo il Modelfile per Ollama...")
            modelfile = self._write_ollama_modelfile(output_dir, gguf_path, request)
            result["modelfile"] = str(modelfile)
            result["ollama_command"] = (
                f'ollama create {request.get("model_name") or self._model_slug()} '
                f'-f "{modelfile}"'
            )

        if not bool(self.conversion_config.get("keep_intermediate", False)):
            # Il modello fp16 intermedio pesa decine di GB e non serve piu'.
            shutil.rmtree(merged_dir, ignore_errors=True)

        self._write_manifest(output_dir, sources, base_model, export_format, request)
        return result

    def _run_gguf_conversion(
        self, merged_dir: Path, output_dir: Path, quantization: str,
        request: Dict[str, Any],
    ) -> Path:
        """Converte in GGUF con gli strumenti di llama.cpp.

        Si passa sempre dagli script di llama.cpp, anche quando Unsloth e'
        installato: la sua `save_pretrained_gguf` chiama comunque llama.cpp,
        e farlo direttamente rende visibile quale eseguibile viene usato e
        cosa manca quando fallisce.
        """
        model_slug = request.get("model_name") or self._model_slug()

        convert_script = self._find_llama_cpp_script()
        if convert_script is None:
            raise RuntimeError(
                "Strumenti llama.cpp non trovati. Per esportare in GGUF:\n"
                "  git clone https://github.com/ggerganov/llama.cpp\n"
                "  cmake -B build llama.cpp && cmake --build build --config Release\n"
                "poi indicare il percorso in conversion.llama_cpp_path "
                "(o nella variabile d'ambiente LLAMA_CPP_PATH)."
            )

        fp16_path = output_dir / f"{model_slug}-f16.gguf"

        self.progress_update.emit(60, "Genero il GGUF a 16 bit...")
        self._run_command(
            [sys.executable, str(convert_script), str(merged_dir),
             "--outfile", str(fp16_path), "--outtype", "f16"],
            "Conversione in GGUF",
        )

        if quantization == "f16":
            return fp16_path

        quantize_binary = self._find_quantize_binary()
        if quantize_binary is None:
            self.error_occurred.emit(
                "Eseguibile di quantizzazione llama.cpp non trovato: "
                f"resta il file a 16 bit ({fp16_path.name})."
            )
            return fp16_path

        quantized_path = output_dir / f"{model_slug}-{quantization}.gguf"
        self.progress_update.emit(75, f"Quantizzo a {quantization}...")
        self._run_command(
            [str(quantize_binary), str(fp16_path), str(quantized_path), quantization],
            f"Quantizzazione {quantization}",
        )

        if quantized_path.exists() and not bool(
            self.conversion_config.get("keep_intermediate", False)
        ):
            fp16_path.unlink(missing_ok=True)

        return quantized_path

    def _write_ollama_modelfile(
        self, output_dir: Path, gguf_path: Path, request: Dict[str, Any]
    ) -> Path:
        """Modelfile col prompt di sistema del personaggio gia' incorporato."""
        system_prompt = request.get("system_prompt") or self._load_system_prompt()
        training = self.config.get("training", {})

        lines = [
            f'FROM ./{gguf_path.name}',
            "",
            f'PARAMETER temperature {self.conversion_config.get("temperature", 0.8)}',
            f'PARAMETER num_ctx {training.get("max_seq_length", 4096)}',
            'PARAMETER repeat_penalty 1.1',
        ]

        if system_prompt:
            # Le virgolette triple permettono un prompt multiriga senza escape.
            lines += ["", 'SYSTEM """', system_prompt.strip(), '"""']

        author = self.config.get("persona", {}).get("author_name", "")
        if author:
            lines += ["", f'# Personaggio: {author}']

        modelfile = output_dir / "Modelfile"
        modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return modelfile

    # ----------------------------------------------------------- strumenti

    def _find_llama_cpp_script(self) -> Optional[Path]:
        for root in self._llama_cpp_roots():
            for name in GGUF_CONVERT_SCRIPTS:
                candidate = root / name
                if candidate.exists():
                    return candidate
        return None

    def _find_quantize_binary(self) -> Optional[Path]:
        for root in self._llama_cpp_roots():
            for name in GGUF_QUANTIZE_BINARIES:
                for candidate in (root / name, root / "build" / "bin" / name):
                    if candidate.exists():
                        return candidate

        for name in GGUF_QUANTIZE_BINARIES:
            found = shutil.which(name)
            if found:
                return Path(found)
        return None

    def _llama_cpp_roots(self) -> List[Path]:
        """Percorsi in cui cercare llama.cpp, dal piu' esplicito al piu' generico."""
        import os

        roots: List[Path] = []
        configured = self.conversion_config.get("llama_cpp_path")
        if configured:
            roots.append(Path(configured))

        env_path = os.environ.get("LLAMA_CPP_PATH")
        if env_path:
            roots.append(Path(env_path))

        roots += [
            self.project_dir.parent / "llama.cpp",
            Path.home() / "llama.cpp",
            Path.cwd() / "llama.cpp",
            # Unsloth Studio scarica una propria build di llama.cpp: se e'
            # installato, l'export GGUF funziona senza installare altro.
            Path.home() / ".unsloth" / "llama.cpp",
        ]
        return [r for r in roots if r.exists()]

    def _run_command(self, command: List[str], description: str) -> None:
        logger.info(f"{description}: {' '.join(command)}")
        try:
            process = subprocess.run(
                command, capture_output=True, text=True,
                timeout=int(self.conversion_config.get("command_timeout", 7200)),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{description}: tempo scaduto") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(f"{description}: comando non trovato ({command[0]})") from exc

        if process.returncode != 0:
            tail = (process.stderr or process.stdout or "").strip().splitlines()[-15:]
            raise RuntimeError(
                f"{description} fallita (codice {process.returncode}):\n" + "\n".join(tail)
            )

    # ------------------------------------------------------------ supporto

    @staticmethod
    def _build_sources(paths: List[Path], weights: Optional[List[float]]) -> List[AdapterSource]:
        sources: List[AdapterSource] = []
        for index, path in enumerate(paths):
            weight = 1.0
            if weights and index < len(weights):
                weight = float(weights[index])
            sources.append(AdapterSource(path=str(path), weight=weight))
        return sources

    def _infer_base_model(self, adapter_paths: List[Path]) -> str:
        """Modello base letto dall'adapter: evita di doverlo ridigitare."""
        for path in adapter_paths:
            info = read_adapter(path)
            if info and info.base_model:
                return info.base_model
        return str(self.config.get("training", {}).get("base_model", ""))

    def _load_for_merge(self, base_model: str, load_in_4bit: bool):
        """Carica il modello base per la fusione, con o senza Unsloth."""
        from ..utils.tls import prepare_model_download

        prepare_model_download()

        max_seq_length = int(self.config.get("training", {}).get("max_seq_length", 4096))

        try:
            from unsloth import FastLanguageModel

            return FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
        except ImportError:
            logger.info("Unsloth non disponibile: uso transformers.")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # La fusione richiede pesi non quantizzati: fondere un adapter su pesi
        # a 4 bit degrada il risultato in modo evitabile.
        from ..stage4_training.training_stage import _dtype_kwarg

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            **_dtype_kwarg(dtype),
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        return model, tokenizer

    def _verify_merged(self, output_dir: Path) -> None:
        """Controlla che nei pesi salvati non resti la struttura dell'adapter.

        Un modello fuso ha i nomi dei tensori del modello originale. Se
        compaiono `base_layer` o `lora_A`/`lora_B`, la fusione non e'
        avvenuta: il file e' comunque valido e si carica, ma non e' un
        modello autonomo, e ogni conversione successiva fallisce con un
        errore che parla di tensori sconosciuti e non dice il perche'.
        """
        try:
            from safetensors import safe_open
        except ImportError:
            return

        shards = sorted(output_dir.glob("*.safetensors"))
        if not shards:
            return

        try:
            with safe_open(str(shards[0]), framework="pt") as handle:
                names = list(handle.keys())
        except Exception as exc:
            self.logger.debug(f"Verifica dei pesi non riuscita: {exc}")
            return

        residual = [n for n in names if "base_layer" in n or "lora_" in n]
        if residual:
            raise RuntimeError(
                "La fusione non ha incorporato l'adapter: nei pesi salvati "
                f"resta la struttura PEFT (es. '{residual[0]}'). "
                "Il modello non e' autonomo e non e' convertibile in GGUF."
            )

    def _check_disk_space(
        self, base_model: str, export_format: ExportFormat, output_dir: Path
    ) -> None:
        """Avvisa prima di iniziare se lo spazio non basta.

        Una conversione che si interrompe a disco pieno dopo venti minuti
        lascia file parziali e nessuna indicazione utile.
        """
        if export_format is ExportFormat.ADAPTER_ONLY:
            return

        billions = estimate_parameters_billions(base_model)
        needed_gb = billions * 2  # 16 bit = 2 byte per parametro
        if export_format.needs_gguf_toolchain:
            needed_gb *= 1.6  # il fp16 intermedio convive col GGUF finale

        try:
            free_gb = shutil.disk_usage(output_dir).free / 1024**3
        except OSError:
            return

        if free_gb < needed_gb:
            message = (
                f"Spazio probabilmente insufficiente: servono circa "
                f"{needed_gb:.0f} GB, liberi {free_gb:.0f} GB su {output_dir.drive or output_dir}."
            )
            self.logger.warning(message)
            self.error_occurred.emit(message)

    def _load_system_prompt(self) -> str:
        prompt_path = self.project_dir / "profiles" / "system_prompt.md"
        if prompt_path.exists():
            try:
                return prompt_path.read_text(encoding="utf-8")
            except OSError as exc:
                self.logger.warning(f"Prompt di sistema illeggibile: {exc}")
        return ""

    def _model_slug(self) -> str:
        author = self.config.get("persona", {}).get("author_name", "") or "persona"
        cleaned = "".join(c if c.isalnum() else "-" for c in author.lower())
        return "-".join(part for part in cleaned.split("-") if part) or "persona"

    def _write_manifest(
        self, output_dir: Path, sources: List[AdapterSource], base_model: str,
        export_format: ExportFormat, request: Dict[str, Any],
    ) -> None:
        manifest = {
            "created_at": datetime.now().isoformat(),
            "format": export_format.value,
            "base_model": base_model,
            "author_name": self.config.get("persona", {}).get("author_name", ""),
            "adapters": [
                {"name": s.name, "path": str(s.path), "weight": s.weight} for s in sources
            ],
            "merge_strategy": request.get("merge_strategy") if len(sources) > 1 else None,
            "quantization": request.get("quantization"),
        }
        (output_dir / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
