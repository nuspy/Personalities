"""Interfaccia a riga di comando: la pipeline senza intervento manuale.

Prima l'unico modo di eseguire la pipeline era la finestra Qt, con quattro
pulsanti da premere nell'ordine giusto: non automatizzabile, non
programmabile, non eseguibile su un server.

    # tutto in automatico, partendo dal solo nome
    python -m historical_persona_pipeline.cli build "Giulio Cesare" \\
        --era "Repubblica romana, I secolo a.C."

    # con documenti propri, senza ricerca online
    python -m historical_persona_pipeline.cli build "Marco Aurelio" \\
        --files ./corpus/*.pdf --no-research

    # solo il profilo, senza generare il dataset (nessun LLM richiesto)
    python -m historical_persona_pipeline.cli build "Cicerone" --until analyze

Gli stadi sono cumulativi: `--until` decide dove fermarsi, e ogni stadio
salva su disco, cosi' un'esecuzione interrotta si riprende da dove era.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_loader import load_config
from .paths import PROJECTS_DIR
from .pipeline.data_models import CompleteStyleProfile, IngestionResult

logger = logging.getLogger("persona")

STAGES = ("research", "ingest", "analyze", "dataset", "train")


# --------------------------------------------------------------------------
# Esecuzione
# --------------------------------------------------------------------------

class PipelineRunner:
    """Esegue gli stadi in sequenza riportando l'avanzamento sul terminale."""

    def __init__(self, config: Dict[str, Any], project_dir: Path, quiet: bool = False):
        self.config = config
        self.project_dir = project_dir
        self.quiet = quiet
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self, stage) -> None:
        """Riporta i segnali Qt dello stadio sul terminale.

        Gli stadi ereditano da QObject per la GUI: i segnali funzionano anche
        senza applicazione Qt, quindi la CLI li usa senza avviare interfacce.
        """
        if self.quiet:
            return
        stage.progress_update.connect(
            lambda pct, msg: print(f"  [{pct:3d}%] {msg}", flush=True)
        )
        stage.error_occurred.connect(
            lambda msg: print(f"  [ATTENZIONE] {msg}", file=sys.stderr, flush=True)
        )

    def _announce(self, title: str) -> None:
        if not self.quiet:
            print(f"\n=== {title} ===", flush=True)

    # ----------------------------------------------------------------- stadi

    def research(self, author: str, era: str, languages: Optional[List[str]]) -> List[Path]:
        from .pipeline.stage0_research.research_stage import ResearchStage

        self._announce(f"Ricerca online: {author}")
        stage = ResearchStage(self.config, self.project_dir)
        self._connect(stage)
        return stage.run({
            "author_name": author,
            "era": era,
            "languages": languages,
        })

    def ingest(self, files: List[Path]) -> IngestionResult:
        from .pipeline.stage1_ingestion.ingestion_stage import IngestionStage

        self._announce(f"Ingestione: {len(files)} file")
        stage = IngestionStage(self.config)
        self._connect(stage)
        result = stage.run(files)

        (self.project_dir / "ingestion.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return result

    def analyze(self, ingestion_result: IngestionResult) -> CompleteStyleProfile:
        from .pipeline.stage2_analysis.analysis_stage import AnalysisStage

        self._announce("Analisi linguistica e costruzione del profilo")
        stage = AnalysisStage(self.config, self.project_dir)
        self._connect(stage)
        return stage.run(ingestion_result)

    def dataset(self, profile: CompleteStyleProfile, ingestion_result: IngestionResult):
        from .pipeline.stage3_dataset.dataset_stage import DatasetStage

        self._announce("Generazione del dataset di addestramento")
        stage = DatasetStage(self.config, self.project_dir)
        self._connect(stage)
        return stage.run({"profile": profile, "ingestion_result": ingestion_result})

    def train(self, plan=None):
        from .pipeline.stage4_training.training_stage import TrainingStage
        from .pipeline.stage4_training.training_modes import TrainingPlan

        plan = plan or TrainingPlan.from_config(self.config)
        self._announce(f"Addestramento — {plan.mode.label}")

        dataset_path = self.project_dir / "datasets" / "train_dataset.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

        stage = TrainingStage(self.config, self.project_dir)
        self._connect(stage)
        return stage.run(dataset_path, plan)

    def convert(self, request):
        from .pipeline.conversion.conversion_stage import ConversionStage

        self._announce(f"Conversione — {request.get('format')}")
        stage = ConversionStage(self.config, self.project_dir)
        self._connect(stage)
        return stage.run(request)


# --------------------------------------------------------------------------
# Comandi
# --------------------------------------------------------------------------

def command_build(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)

    config["persona"]["author_name"] = args.author
    if args.era:
        config["persona"]["era"] = args.era
    if args.value_dict:
        config["persona"]["value_dict"] = args.value_dict
    if args.conversations:
        config["dataset"]["num_conversations"] = args.conversations
    if args.llm_url:
        config["dataset"]["llm_base_url"] = args.llm_url
    if args.llm_model:
        config["dataset"]["llm_model_name"] = args.llm_model
    if args.no_research:
        config["research"]["enabled"] = False

    project_dir = Path(args.output) if args.output else PROJECTS_DIR / _slug(args.author)
    runner = PipelineRunner(config, project_dir, quiet=args.quiet)

    print(f"Progetto: {project_dir}")
    stop_after = args.until
    stop_index = STAGES.index(stop_after)

    files: List[Path] = []

    # --- ricerca -----------------------------------------------------------
    if not args.no_research:
        try:
            files.extend(runner.research(args.author, args.era or "", args.languages))
        except Exception as exc:
            logger.error(f"Ricerca online fallita: {exc}")
            if not args.files:
                print(
                    "\nLa ricerca online non ha prodotto nulla e non sono stati "
                    "indicati file locali (--files). Interrotto.",
                    file=sys.stderr,
                )
                return 1

    # --- file locali -------------------------------------------------------
    for pattern in args.files or []:
        matched = _expand(pattern)
        if not matched:
            print(f"Nessun file corrisponde a '{pattern}'", file=sys.stderr)
        files.extend(matched)

    if stop_index == STAGES.index("research"):
        print(f"\nCompletato. {len(files)} documenti in {project_dir / 'sources'}")
        return 0

    if not files:
        print("Nessun documento da analizzare.", file=sys.stderr)
        return 1

    # --- ingestione --------------------------------------------------------
    ingestion_result = runner.ingest(files)
    _report_ingestion(ingestion_result)

    if stop_index == STAGES.index("ingest"):
        return 0

    # --- analisi -----------------------------------------------------------
    profile = runner.analyze(ingestion_result)
    _report_profile(profile, project_dir)

    if stop_index == STAGES.index("analyze"):
        return 0

    # --- dataset -----------------------------------------------------------
    try:
        dataset = runner.dataset(profile, ingestion_result)
    except Exception as exc:
        print(f"\nGenerazione del dataset fallita: {exc}", file=sys.stderr)
        print(
            "Il profilo e' comunque salvato: "
            f"{project_dir / 'profiles' / 'system_prompt.md'}",
            file=sys.stderr,
        )
        return 1

    print(f"\nDataset: {dataset.total_conversations} conversazioni")
    for category, count in sorted(dataset.type_distribution.items(), key=lambda kv: -kv[1]):
        print(f"  {category:24} {count}")

    if stop_index == STAGES.index("dataset"):
        return 0

    # --- addestramento -----------------------------------------------------
    runner.train(_plan_from_args(args, config))
    return 0


def _plan_from_args(args: argparse.Namespace, config: Dict[str, Any]):
    """Costruisce il piano di addestramento dagli argomenti di riga di comando."""
    from .pipeline.stage4_training.training_modes import (
        AdapterSource, TrainingMode, TrainingPlan,
    )

    plan = TrainingPlan.from_config(config)

    mode = getattr(args, "mode", None)
    if mode:
        plan.mode = TrainingMode(mode)

    from_adapters = getattr(args, "from_adapter", None)
    if from_adapters:
        weights = getattr(args, "adapter_weight", None) or []
        plan.adapters = [
            AdapterSource(
                path=path,
                weight=float(weights[i]) if i < len(weights) else 1.0,
            )
            for i, path in enumerate(from_adapters)
        ]
        # Indicare adapter senza scegliere la modalita' significa quasi sempre
        # volerli impilare: continuare e' una scelta piu' specifica, che va
        # dichiarata.
        if not mode and plan.mode is TrainingMode.LORA_NEW:
            plan.mode = TrainingMode.LORA_STACK

    if getattr(args, "merge_strategy", None):
        plan.merge_strategy = args.merge_strategy
    if getattr(args, "output_name", None):
        plan.output_name = args.output_name
    if getattr(args, "base_model", None):
        plan.base_model = args.base_model

    return plan


def command_train(args: argparse.Namespace) -> int:
    """Addestra usando un dataset gia' generato."""
    config = load_config(Path(args.config) if args.config else None)
    project_dir = Path(args.project)

    if not project_dir.exists():
        print(f"Progetto non trovato: {project_dir}", file=sys.stderr)
        return 1

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    plan = _plan_from_args(args, config)
    problems = plan.validate()
    if problems:
        for problem in problems:
            print(f"Errore: {problem}", file=sys.stderr)
        return 1

    print(f"Modalità : {plan.mode.label}")
    print(f"Base     : {plan.base_model}")
    if plan.adapters:
        print("Partendo da:")
        for adapter in plan.adapters:
            print(f"  - {adapter.name} (peso {adapter.weight:g})")
        if len(plan.adapters) > 1:
            print(f"Fusione  : {plan.merge_strategy}")

    runner = PipelineRunner(config, project_dir, quiet=args.quiet)
    result = runner.train(plan)

    print(f"\nRisultato: {result.get('output_path')}")
    return 0


def command_convert(args: argparse.Namespace) -> int:
    """Converte uno o più adapter in un formato distribuibile."""
    from .pipeline.conversion.formats import ExportFormat

    config = load_config(Path(args.config) if args.config else None)
    project_dir = Path(args.project) if args.project else PROJECTS_DIR

    request: Dict[str, Any] = {
        "adapters": args.adapters,
        "weights": args.weights,
        "format": args.format,
        "merge_strategy": args.merge_strategy,
        "base_model": args.base_model or "",
        "model_name": args.name or "",
        "quantization": args.quantization,
    }
    if args.output:
        request["output_dir"] = args.output

    runner = PipelineRunner(config, project_dir, quiet=args.quiet)
    result = runner.convert(request)

    print(f"\nFormato   : {ExportFormat(result['format']).label}")
    print(f"Percorso  : {result['output_path']}")
    if result.get("gguf_path"):
        print(f"File GGUF : {result['gguf_path']}")
    if result.get("ollama_command"):
        print(f"\nPer importarlo in Ollama:\n  {result['ollama_command']}")
    return 0


def command_adapters(args: argparse.Namespace) -> int:
    """Elenca gli adapter LoRA trovati su disco."""
    from .pipeline.stage4_training.adapter_manager import discover_adapters

    roots = [Path(r) for r in (args.root or [])] or [PROJECTS_DIR]
    adapters = discover_adapters(roots)

    if not adapters:
        print(f"Nessun adapter trovato in: {', '.join(str(r) for r in roots)}")
        return 0

    print(f"{'adapter':28} {'rango':>6} {'MB':>8}  modello base")
    print("-" * 88)
    for adapter in adapters:
        base = adapter.base_model.split("/")[-1] if adapter.base_model else "?"
        print(f"{adapter.name:28} {adapter.rank:>6} {adapter.size_mb:>8.0f}  {base}")
        if args.verbose:
            print(f"{'':28} {adapter.path}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    """Mostra il profilo di un progetto gia' costruito."""
    profile_path = Path(args.project) / "profiles" / "style_profile.json"
    if not profile_path.exists():
        print(f"Profilo non trovato: {profile_path}", file=sys.stderr)
        return 1

    profile = CompleteStyleProfile.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )

    if args.prompt_only:
        print(profile.generated_system_prompt)
        return 0

    _report_profile(profile, Path(args.project), verbose=True)
    return 0


def command_list(args: argparse.Namespace) -> int:
    """Elenca i progetti presenti su disco."""
    root = Path(args.root) if args.root else PROJECTS_DIR
    if not root.exists():
        print(f"Nessun progetto in {root}")
        return 0

    rows = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        profile_path = directory / "profiles" / "style_profile.json"
        if profile_path.exists():
            try:
                data = json.loads(profile_path.read_text(encoding="utf-8"))
                rows.append((
                    directory.name,
                    data.get("author_name", "?"),
                    data.get("corpus_stats", {}).get("total_words", 0),
                    data.get("generated_at", "")[:10],
                ))
                continue
            except Exception:
                pass
        rows.append((directory.name, "(incompleto)", 0, ""))

    if not rows:
        print(f"Nessun progetto in {root}")
        return 0

    print(f"{'progetto':30} {'personaggio':28} {'parole':>10}  creato")
    print("-" * 82)
    for name, author, words, created in rows:
        print(f"{name:30} {author:28} {words:>10,}  {created}".replace(",", "."))
    return 0


# --------------------------------------------------------------------------
# Presentazione
# --------------------------------------------------------------------------

def _report_ingestion(result: IngestionResult) -> None:
    words = sum(s.word_count for s in result.segments)
    print(f"\n  file letti : {result.total_files_processed}")
    print(f"  segmenti   : {result.total_segments}")
    print(f"  parole     : {words:,}".replace(",", "."))
    languages = ", ".join(
        f"{lang.value}={count}" for lang, count in result.language_distribution.items()
    )
    print(f"  lingue     : {languages}")

    if result.errors:
        print(f"  avvisi     : {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"    - {Path(error['file']).name}: {error['error'][:90]}")


def _report_profile(
    profile: CompleteStyleProfile, project_dir: Path, verbose: bool = False
) -> None:
    print(f"\n  personaggio   : {profile.author_name}")
    print(f"  lingua        : {profile.primary_language.value}")
    print(f"  frase media   : {profile.syntax.avg_sentence_length} parole")
    print(f"  prospettiva   : {profile.rhetoric.narrative_perspective}")
    print(f"  registro      : {profile.voice.speech_register}")

    if profile.values.dominant_values:
        print(f"  valori        : {', '.join(profile.values.dominant_values[:5])}")

    terms = [t["term"] for t in profile.vocabulary.distinctive_terms[:8]]
    if terms:
        print(f"  lessico       : {', '.join(terms)}")

    formulas = profile.vocabulary.formulas_and_collocations[:5]
    if formulas:
        print(f"  formule       : {', '.join(formulas)}")

    if verbose:
        openers = [o["opener"] for o in profile.voice.signature_openers[:10]]
        if openers:
            print(f"  incipit       : {', '.join(openers)}")
        if profile.rhetoric.argument_type_distribution:
            ordered = sorted(
                profile.rhetoric.argument_type_distribution.items(), key=lambda kv: -kv[1]
            )
            print("  argomentazione: " + ", ".join(f"{k} {v:.0f}%" for k, v in ordered[:5]))
        print("\n" + "-" * 70)
        print(profile.generated_system_prompt)

    print(f"\n  profilo salvato in {project_dir / 'profiles'}")


# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------

def _slug(name: str) -> str:
    import re

    cleaned = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    return re.sub(r"[\s_-]+", "_", cleaned) or "progetto"


def _expand(pattern: str) -> List[Path]:
    """Espande un percorso, un glob o una cartella in una lista di file."""
    path = Path(pattern)

    if path.is_file():
        return [path]

    if path.is_dir():
        from .pipeline.stage1_ingestion.decoders import supported_extensions

        files: List[Path] = []
        for extension in supported_extensions():
            files.extend(path.rglob(f"*{extension}"))
        return sorted(files)

    # Glob: la parte fissa iniziale fa da radice della ricerca.
    parent = path.parent if str(path.parent) != "." else Path.cwd()
    try:
        return sorted(p for p in parent.glob(path.name) if p.is_file())
    except (ValueError, OSError):
        return []


def _export_formats():
    from .pipeline.conversion.formats import ExportFormat

    return list(ExportFormat)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Opzioni di addestramento, condivise da `build` e `train`."""
    group = parser.add_argument_group("addestramento")
    group.add_argument(
        "--mode",
        choices=["lora_new", "lora_continue", "lora_stack", "full_finetune"],
        help=(
            "lora_new: adapter nuovo · lora_continue: prosegue un adapter · "
            "lora_stack: fonde gli adapter indicati e addestra sopra · "
            "full_finetune: tutti i pesi"
        ),
    )
    group.add_argument(
        "--from-adapter", nargs="+", metavar="PATH",
        help="adapter di partenza; senza --mode implica lora_stack",
    )
    group.add_argument(
        "--adapter-weight", nargs="+", type=float, metavar="P",
        help="peso di ciascun adapter nella fusione (default: 1.0)",
    )
    group.add_argument(
        "--merge-strategy", metavar="NOME",
        help="linear | cat | ties | dare_ties | svd",
    )
    group.add_argument("--base-model", metavar="NOME", help="modello base")
    group.add_argument(
        "--output-name", metavar="NOME",
        help="nome della cartella prodotta sotto <progetto>/output/",
    )


def build_parser() -> argparse.ArgumentParser:
    # Le opzioni comuni stanno in un parser genitore, cosi' `-q` e `-v`
    # funzionano sia prima sia dopo il sottocomando: pretendere l'ordine
    # esatto e' un inciampo gratuito per chi usa lo strumento.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="log dettagliato")
    common.add_argument("-q", "--quiet", action="store_true", help="nessun avanzamento")

    parser = argparse.ArgumentParser(
        prog="persona",
        description="Costruisce la personalita' di un personaggio da documenti e fonti online.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="costruisce un personaggio", parents=[common])
    build.add_argument("author", help="nome del personaggio")
    build.add_argument("--era", help="epoca, es. 'Repubblica romana, I secolo a.C.'")
    build.add_argument("--files", nargs="+", metavar="PATH",
                       help="file, cartelle o pattern da includere")
    build.add_argument("--no-research", action="store_true",
                       help="non cercare online: usa solo i file indicati")
    build.add_argument("--languages", nargs="+", metavar="LANG",
                       help="lingue in cui cercare (default: dedotte dall'epoca)")
    build.add_argument("--until", choices=STAGES, default="dataset",
                       help="ultimo stadio da eseguire (default: dataset)")
    build.add_argument("--conversations", type=int, metavar="N",
                       help="numero di conversazioni da generare")
    build.add_argument("--value-dict", metavar="NOME",
                       help="dizionario di valori da usare (default: automatico)")
    build.add_argument("--llm-url", metavar="URL", help="endpoint LLM")
    build.add_argument("--llm-model", metavar="NOME", help="modello LLM")
    build.add_argument("--output", metavar="DIR", help="cartella di progetto")
    build.add_argument("--config", metavar="FILE", help="file di configurazione")
    _add_training_arguments(build)
    build.set_defaults(func=command_build)

    train = subparsers.add_parser(
        "train", help="addestra su un dataset già generato", parents=[common]
    )
    train.add_argument("project", help="cartella del progetto")
    train.add_argument("--epochs", type=int, metavar="N", help="numero di epoche")
    train.add_argument("--config", metavar="FILE", help="file di configurazione")
    _add_training_arguments(train)
    train.set_defaults(func=command_train)

    convert = subparsers.add_parser(
        "convert", help="converte adapter in modello distribuibile", parents=[common]
    )
    convert.add_argument("adapters", nargs="+", metavar="ADAPTER",
                         help="cartelle degli adapter da convertire")
    convert.add_argument("--format", default="merged_16bit",
                         choices=[f.value for f in _export_formats()],
                         help="formato di destinazione (default: merged_16bit)")
    convert.add_argument("--quantization", default="q4_k_m", metavar="LIVELLO",
                         help="livello GGUF: f16, q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k")
    convert.add_argument("--weights", nargs="+", type=float, metavar="P",
                         help="peso di ciascun adapter nella fusione")
    convert.add_argument("--merge-strategy", default="linear",
                         help="linear | cat | ties | dare_ties | svd")
    convert.add_argument("--base-model", metavar="NOME",
                         help="modello base (default: dedotto dall'adapter)")
    convert.add_argument("--name", metavar="NOME", help="nome del modello prodotto")
    convert.add_argument("--output", metavar="DIR", help="cartella di destinazione")
    convert.add_argument("--project", metavar="DIR",
                         help="progetto da cui prendere il prompt di sistema")
    convert.add_argument("--config", metavar="FILE", help="file di configurazione")
    convert.set_defaults(func=command_convert)

    adapters = subparsers.add_parser(
        "adapters", help="elenca gli adapter disponibili", parents=[common]
    )
    adapters.add_argument("--root", nargs="+", metavar="DIR",
                          help="cartelle in cui cercare (default: progetti)")
    adapters.set_defaults(func=command_adapters)

    inspect = subparsers.add_parser(
        "inspect", help="mostra un profilo gia' costruito", parents=[common]
    )
    inspect.add_argument("project", help="cartella del progetto")
    inspect.add_argument("--prompt-only", action="store_true",
                         help="stampa solo il prompt di sistema")
    inspect.set_defaults(func=command_inspect)

    listing = subparsers.add_parser("list", help="elenca i progetti", parents=[common])
    listing.add_argument("--root", metavar="DIR", help="cartella dei progetti")
    listing.set_defaults(func=command_list)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    started = datetime.now()
    try:
        exit_code = args.func(args)
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Esecuzione fallita")
        print(f"\nErrore: {exc}", file=sys.stderr)
        return 1

    if not args.quiet and exit_code == 0:
        elapsed = (datetime.now() - started).total_seconds()
        print(f"\nCompletato in {elapsed:.0f}s")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
