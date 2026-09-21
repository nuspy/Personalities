"""Test di modalita' di addestramento, gestione adapter e conversione.

Nessun test carica un modello: servirebbero GPU e decine di GB. Qui si
verifica la logica che decide *cosa* fare — scoperta degli adapter, controlli
di compatibilita', validazione dei piani, costruzione delle richieste di
conversione — cioe' la parte che sbaglia in silenzio quando e' sbagliata.
"""
from __future__ import annotations

import json

import pytest

from historical_persona_pipeline.config_loader import load_config
from historical_persona_pipeline.pipeline.conversion.formats import (
    DEFAULT_QUANTIZATION, QUANTIZATIONS, ExportFormat, estimate_parameters_billions,
)
from historical_persona_pipeline.pipeline.stage4_training.adapter_manager import (
    check_compatibility, discover_adapters, read_adapter,
)
from historical_persona_pipeline.pipeline.stage4_training.training_modes import (
    MERGE_STRATEGIES, AdapterSource, TrainingMode, TrainingPlan,
)


def make_adapter(
    directory,
    name: str = "adapter",
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct",
    rank: int = 64,
    target_modules=("q_proj", "v_proj"),
):
    """Crea sul disco una cartella con la forma di un adapter PEFT."""
    adapter_dir = directory / name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({
            "base_model_name_or_path": base_model,
            "r": rank,
            "lora_alpha": rank * 2,
            "target_modules": list(target_modules),
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\0" * 2048)
    return adapter_dir


class TestAdapterDiscovery:
    def test_adapter_letto_correttamente(self, tmp_path):
        path = make_adapter(tmp_path, "cesare", rank=32)
        info = read_adapter(path)

        assert info is not None
        assert info.name == "cesare"
        assert info.rank == 32
        assert info.alpha == 64
        assert info.is_valid

    def test_cartella_qualsiasi_non_e_un_adapter(self, tmp_path):
        (tmp_path / "vuota").mkdir()
        assert read_adapter(tmp_path / "vuota") is None

    def test_adapter_senza_pesi_non_e_valido(self, tmp_path):
        path = make_adapter(tmp_path, "monco")
        (path / "adapter_model.safetensors").unlink()
        info = read_adapter(path)
        assert info is not None and not info.is_valid

    def test_scoperta_ricorsiva(self, tmp_path):
        make_adapter(tmp_path / "progetti" / "a" / "output", "lora_adapters")
        make_adapter(tmp_path / "progetti" / "b" / "output", "lora_adapters")

        found = discover_adapters([tmp_path])
        assert len(found) == 2

    def test_scoperta_non_duplica(self, tmp_path):
        make_adapter(tmp_path / "output", "lora")
        found = discover_adapters([tmp_path, tmp_path])
        assert len(found) == 1

    def test_cartella_inesistente_non_solleva(self, tmp_path):
        assert discover_adapters([tmp_path / "non_esiste"]) == []


class TestCompatibility:
    def test_adapter_dello_stesso_modello_sono_compatibili(self, tmp_path):
        infos = [
            read_adapter(make_adapter(tmp_path, "a")),
            read_adapter(make_adapter(tmp_path, "b")),
        ]
        errors, _ = check_compatibility(infos)
        assert errors == []

    def test_modelli_base_diversi_sono_un_errore(self, tmp_path):
        """Fondere un adapter Llama con uno Mistral non fallisce: produce
        pesi incoerenti. Va fermato prima."""
        infos = [
            read_adapter(make_adapter(tmp_path, "llama", base_model="meta-llama/Llama-3.1-8B")),
            read_adapter(make_adapter(tmp_path, "mistral", base_model="mistralai/Mistral-7B")),
        ]
        errors, _ = check_compatibility(infos)
        assert errors
        assert "modelli base diversi" in errors[0].lower()

    def test_quantizzazione_nel_nome_non_conta_come_differenza(self, tmp_path):
        """`...-bnb-4bit` e' lo stesso modello, solo distribuito quantizzato."""
        infos = [
            read_adapter(make_adapter(
                tmp_path, "a", base_model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
            ))
        ]
        _, warnings = check_compatibility(infos, "meta-llama/Meta-Llama-3.1-8B-Instruct")
        assert not any("non coincide" in w for w in warnings)

    def test_ranghi_diversi_producono_un_avviso(self, tmp_path):
        infos = [
            read_adapter(make_adapter(tmp_path, "r16", rank=16)),
            read_adapter(make_adapter(tmp_path, "r64", rank=64)),
        ]
        errors, warnings = check_compatibility(infos)
        assert errors == []
        assert any("svd" in w.lower() for w in warnings)

    def test_moduli_diversi_producono_un_avviso(self, tmp_path):
        infos = [
            read_adapter(make_adapter(tmp_path, "a", target_modules=("q_proj",))),
            read_adapter(make_adapter(tmp_path, "b", target_modules=("q_proj", "v_proj"))),
        ]
        _, warnings = check_compatibility(infos)
        assert any("moduli" in w.lower() for w in warnings)

    def test_nessun_adapter_nessun_problema(self):
        assert check_compatibility([]) == ([], [])


class TestTrainingPlan:
    def test_lora_nuova_non_richiede_adapter(self):
        assert TrainingPlan(mode=TrainingMode.LORA_NEW).validate() == []

    def test_continue_senza_adapter_e_invalido(self):
        problems = TrainingPlan(mode=TrainingMode.LORA_CONTINUE).validate()
        assert problems and "richiede almeno un adapter" in problems[0]

    def test_continue_con_piu_adapter_e_invalido(self):
        """Continuare significa modificare *un* adapter: con due non e' definito."""
        plan = TrainingPlan(
            mode=TrainingMode.LORA_CONTINUE,
            adapters=[AdapterSource("/a"), AdapterSource("/b")],
        )
        problems = plan.validate()
        assert problems and "un solo adapter" in problems[0]

    def test_stack_accetta_piu_adapter(self):
        plan = TrainingPlan(
            mode=TrainingMode.LORA_STACK,
            adapters=[AdapterSource("/a"), AdapterSource("/b")],
        )
        assert plan.validate() == []

    def test_strategia_sconosciuta_e_invalida(self):
        plan = TrainingPlan(mode=TrainingMode.LORA_NEW, merge_strategy="inventata")
        assert any("Strategia di fusione" in p for p in plan.validate())

    def test_peso_non_positivo_e_invalido(self):
        plan = TrainingPlan(
            mode=TrainingMode.LORA_STACK,
            adapters=[AdapterSource("/a", weight=0.0)],
        )
        assert any("Peso non valido" in p for p in plan.validate())

    def test_piano_letto_dalla_configurazione(self):
        config = load_config()
        config["training"]["mode"] = "lora_stack"
        config["training"]["start_from_adapters"] = [
            {"path": "/percorso/a", "weight": 0.7, "name": "stile"},
            "/percorso/b",
        ]

        plan = TrainingPlan.from_config(config)
        assert plan.mode is TrainingMode.LORA_STACK
        assert len(plan.adapters) == 2
        assert plan.adapters[0].weight == 0.7
        assert plan.adapters[1].name == "b"  # dedotto dal percorso

    def test_modalita_sconosciuta_ripiega_sul_default(self):
        config = load_config()
        config["training"]["mode"] = "modalita_inesistente"
        assert TrainingPlan.from_config(config).mode is TrainingMode.LORA_NEW

    def test_solo_continue_e_stack_richiedono_adapter(self):
        assert TrainingMode.LORA_CONTINUE.requires_adapters
        assert TrainingMode.LORA_STACK.requires_adapters
        assert not TrainingMode.LORA_NEW.requires_adapters
        assert not TrainingMode.FULL_FINETUNE.requires_adapters

    def test_solo_il_finetuning_completo_non_produce_adapter(self):
        assert not TrainingMode.FULL_FINETUNE.produces_adapter
        assert TrainingMode.LORA_NEW.produces_adapter

    def test_ogni_modalita_ha_etichetta_e_descrizione(self):
        for mode in TrainingMode:
            assert mode.label and mode.description

    def test_ogni_strategia_e_documentata(self):
        for name, description in MERGE_STRATEGIES.items():
            assert description, f"strategia '{name}' senza descrizione"


class TestExportFormats:
    def test_ogni_formato_ha_etichetta_e_descrizione(self):
        for export_format in ExportFormat:
            assert export_format.label and export_format.description

    def test_solo_gguf_e_ollama_richiedono_llama_cpp(self):
        assert ExportFormat.GGUF.needs_gguf_toolchain
        assert ExportFormat.OLLAMA.needs_gguf_toolchain
        assert not ExportFormat.MERGED_16BIT.needs_gguf_toolchain
        assert not ExportFormat.ADAPTER_ONLY.needs_gguf_toolchain

    def test_dimensione_stimata_cresce_coi_bit(self):
        sizes = [QUANTIZATIONS[q].estimated_size_gb(8) for q in ("q2_k", "q4_k_m", "q8_0", "f16")]
        assert sizes == sorted(sizes)

    def test_quantizzazione_predefinita_esiste(self):
        assert DEFAULT_QUANTIZATION in QUANTIZATIONS

    @pytest.mark.parametrize("name,expected", [
        ("unsloth/Meta-Llama-3.1-8B-Instruct", 8.0),
        ("mistralai/Mistral-7B-v0.1", 7.0),
        ("Qwen/Qwen2.5-14B", 14.0),
        ("google/gemma-2-2b-it", 2.0),
    ])
    def test_dimensione_dedotta_dal_nome(self, name, expected):
        assert estimate_parameters_billions(name) == expected

    def test_nome_senza_taglia_usa_un_valore_prudente(self):
        assert estimate_parameters_billions("un/modello-senza-taglia") == 8.0


class TestConversionStage:
    """Solo la parte che non richiede di caricare un modello."""

    @pytest.fixture
    def stage(self, tmp_path):
        from historical_persona_pipeline.pipeline.conversion.conversion_stage import (
            ConversionStage,
        )

        config = load_config()
        config["persona"]["author_name"] = "Gaio Giulio Cesare"
        return ConversionStage(config, tmp_path)

    def test_modello_base_dedotto_dall_adapter(self, stage, tmp_path):
        """Evita di dover ridigitare il nome del modello: e' gia' nell'adapter."""
        path = make_adapter(tmp_path, "cesare", base_model="meta-llama/Llama-3.1-8B")
        assert stage._infer_base_model([path]) == "meta-llama/Llama-3.1-8B"

    def test_senza_adapter_solleva(self, stage):
        with pytest.raises(ValueError, match="Nessun adapter"):
            stage.run({"adapters": [], "format": "merged_16bit"})

    def test_quantizzazione_sconosciuta_solleva(self, stage, tmp_path):
        path = make_adapter(tmp_path, "cesare")
        with pytest.raises(ValueError, match="Quantizzazione sconosciuta"):
            stage.run({
                "adapters": [str(path)],
                "format": "gguf",
                "quantization": "q99_impossibile",
            })

    def test_nome_modello_derivato_dal_personaggio(self, stage):
        assert stage._model_slug() == "gaio-giulio-cesare"

    def test_modelfile_contiene_il_prompt(self, stage, tmp_path):
        prompt_dir = tmp_path / "profiles"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "system_prompt.md").write_text(
            "Sei Gaio Giulio Cesare.", encoding="utf-8"
        )

        output_dir = tmp_path / "export"
        output_dir.mkdir()
        modelfile = stage._write_ollama_modelfile(
            output_dir, output_dir / "modello-q4_k_m.gguf", {}
        )

        content = modelfile.read_text(encoding="utf-8")
        assert "FROM ./modello-q4_k_m.gguf" in content
        assert "Sei Gaio Giulio Cesare." in content
        assert "PARAMETER num_ctx" in content

    def test_prompt_esplicitamente_vuoto_non_finisce_nel_modelfile(self, stage, tmp_path):
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        modelfile = stage._write_ollama_modelfile(
            output_dir, output_dir / "m.gguf", {"system_prompt": ""}
        )
        assert "SYSTEM" not in modelfile.read_text(encoding="utf-8")

    def test_sorgenti_con_pesi(self, stage):
        from pathlib import Path

        sources = stage._build_sources([Path("/a"), Path("/b")], [0.7, 0.3])
        assert [s.weight for s in sources] == [0.7, 0.3]

    def test_sorgenti_senza_pesi_usano_uno(self, stage):
        from pathlib import Path

        sources = stage._build_sources([Path("/a"), Path("/b")], None)
        assert all(s.weight == 1.0 for s in sources)


class TestChatTemplateFormatting:
    """Bug trovato eseguendo il training reale, non dai test.

    I template di `transformers` leggono `role`/`content`. Passando il
    formato ShareGPT (`from`/`value`) non si ottiene un errore: il template
    non trova i campi che cerca, scarta l'intera conversazione e restituisce
    il prompt di sistema di default del modello. L'addestramento gira
    regolarmente, la loss scende, e l'adapter non contiene nulla del
    personaggio — ce ne si accorge solo generando.
    """

    def test_ruoli_sharegpt_convertiti(self):
        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            _to_standard_roles,
        )

        turns = _to_standard_roles([
            {"from": "system", "value": "Sei Cesare."},
            {"from": "human", "value": "Chi sei?"},
            {"from": "gpt", "value": "Caesar sum."},
        ])
        assert turns == [
            {"role": "system", "content": "Sei Cesare."},
            {"role": "user", "content": "Chi sei?"},
            {"role": "assistant", "content": "Caesar sum."},
        ]

    def test_ruoli_gia_standard_restano_invariati(self):
        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            _to_standard_roles,
        )

        original = [{"role": "user", "content": "ciao"}]
        assert _to_standard_roles(original) == original

    def test_ruolo_sconosciuto_diventa_user(self):
        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            _to_standard_roles,
        )

        turns = _to_standard_roles([{"from": "qualcosa", "value": "testo"}])
        assert turns[0]["role"] == "user"

    def test_template_reale_conserva_il_contenuto(self):
        """Verifica con un template Jinja come quelli veri dei modelli."""
        from jinja2 import Template

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            _to_standard_roles,
        )

        # Forma equivalente al template ChatML usato da Qwen e molti altri.
        template = Template(
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
            "{{ m['content'] }}<|im_end|>\n{% endfor %}"
        )
        sharegpt = [
            {"from": "system", "value": "Sei Cesare."},
            {"from": "human", "value": "Chi sei?"},
            {"from": "gpt", "value": "Caesar sum."},
        ]

        # Senza conversione il contenuto sparisce.
        raw = template.render(messages=sharegpt)
        assert "Caesar sum." not in raw

        # Con la conversione arriva a destinazione.
        converted = template.render(messages=_to_standard_roles(sharegpt))
        assert "Caesar sum." in converted
        assert "Sei Cesare." in converted


class TestFormattingVerification:
    """Il controllo che impedisce al bug precedente di ripresentarsi."""

    @pytest.fixture
    def stage(self, tmp_path):
        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        return TrainingStage(load_config(), tmp_path)

    def test_testo_corretto_passa(self, stage):
        dataset = [{
            "text": "<|im_start|>assistant\nCaesar in Galliam profectus est.<|im_end|>",
            "conversations": [{"from": "gpt", "value": "Caesar in Galliam profectus est."}],
        }]
        stage._verify_formatting(_FakeDataset(dataset))  # nessuna eccezione

    def test_contenuto_mancante_solleva(self, stage):
        """Il caso reale: il template ha scartato la conversazione."""
        dataset = [{
            "text": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.<|im_end|>",
            "conversations": [{"from": "gpt", "value": "Caesar in Galliam profectus est."}],
        }]
        with pytest.raises(ValueError, match="non contiene le conversazioni"):
            stage._verify_formatting(_FakeDataset(dataset))

    def test_testo_vuoto_solleva(self, stage):
        with pytest.raises(ValueError, match="testo vuoto"):
            stage._verify_formatting(_FakeDataset([{"text": "   ", "conversations": []}]))

    def test_dataset_vuoto_solleva(self, stage):
        with pytest.raises(ValueError, match="vuoto"):
            stage._verify_formatting(_FakeDataset([]))


class _FakeDataset(list):
    """Minimo indispensabile dell'interfaccia di `datasets.Dataset`."""

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in list.__iter__(self)]
        return list.__getitem__(self, key)


class TestDtypeKwarg:
    """`torch_dtype` e' stato rinominato `dtype` in transformers 4.56."""

    def test_nome_corretto_per_la_versione_installata(self):
        import transformers

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            _dtype_kwarg,
        )

        version = tuple(int(p) for p in transformers.__version__.split(".")[:2] if p.isdigit())
        expected = "dtype" if version >= (4, 56) else "torch_dtype"
        assert list(_dtype_kwarg("float16")) == [expected]


class TestUnslothShareGPTMapping:
    """Il parametro `mapping` di Unsloth non converte i dati.

    `get_chat_template(tokenizer, mapping={"role": "from", ...})` sembra fatto
    apposta per leggere i turni ShareGPT, e il codice ci si era affidato
    saltando la conversione sul ramo Unsloth. Verificandolo su Unsloth 2026.9
    si vede che i turni arrivano al template senza essere tradotti: restano
    solo quelli il cui ruolo coincide gia' con lo standard (`system`), mentre
    `human` e `gpt` spariscono. Lo stesso fallimento silenzioso del ramo
    transformers, in un punto in cui sembrava impossibile.

    Oggi la conversione e' incondizionata: questi test fissano il fatto che
    non dipenda dal backend.
    """

    def test_conversione_applicata_a_prescindere_dal_backend(self):
        """Il codice non deve piu' contenere un ramo per saltare la conversione."""
        import inspect

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        source = inspect.getsource(TrainingStage._prepare_datasets)
        assert "_to_standard_roles" in source
        assert "needs_conversion" not in source

    def test_nessun_mapping_passato_a_get_chat_template(self):
        """Passare `mapping` suggerirebbe una conversione che non avviene."""
        import inspect

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        source = inspect.getsource(TrainingStage._apply_chat_template)
        assert "mapping=" not in source

    def test_unsloth_importato_prima_delle_librerie_che_patcha(self):
        """Unsloth applica le ottimizzazioni con una patch a transformers,
        trl e peft: importato dopo di loro, non ha effetto."""
        import inspect

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        source = inspect.getsource(TrainingStage.run)
        unsloth_at = source.find("import unsloth")
        trl_at = source.find("from trl import")
        assert unsloth_at != -1, "unsloth non importato in run()"
        assert unsloth_at < trl_at, "unsloth deve precedere trl"


class TestUnslothBranchRegressions:
    """Difetti emersi solo eseguendo il ciclo con Unsloth installato."""

    def test_ensure_trainable_abilita_i_gradienti_sugli_input(self):
        """Col gradient checkpointing le attivazioni sono ricalcolate: se
        l'ingresso del blocco non richiede gradiente il grafo si spezza, e
        l'errore — «element 0 of tensors does not require grad» — non nomina
        ne' gli adapter ne' il checkpointing."""
        import inspect

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        source = inspect.getsource(TrainingStage._ensure_trainable)
        assert "enable_input_require_grads" in source

    def test_ensure_trainable_riporta_unsloth_in_addestramento(self):
        import inspect

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        source = inspect.getsource(TrainingStage._ensure_trainable)
        assert "for_training" in source

    def test_fallback_unsloth_non_limitato_a_importerror(self):
        """Unsloth ricompila i moduli del modello e la compilazione puo'
        fallire per combinazioni di versioni. Non e' un motivo per fermare
        l'addestramento: transformers carica lo stesso modello."""
        import inspect

        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        source = inspect.getsource(TrainingStage._load_base_model)
        assert "_report_unsloth_fallback" in source
        assert "except Exception" in source

    def test_fusione_preferisce_merge_and_unload(self):
        """`save_pretrained_merged` su un modello avvolto da PEFT lascia nei
        pesi la struttura dell'adapter: il salvataggio riesce e il problema
        emerge solo alla conversione GGUF."""
        import inspect

        from historical_persona_pipeline.pipeline.conversion.conversion_stage import (
            ConversionStage,
        )

        source = inspect.getsource(ConversionStage._export_merged)
        merge_at = source.find("merge_and_unload")
        save_merged_at = source.find("save_pretrained_merged")
        assert merge_at != -1
        assert merge_at < save_merged_at, "merge_and_unload deve avere la precedenza"


class TestMergedVerification:
    """Il controllo che impedisce di consegnare un 'modello fuso' che non lo e'."""

    @pytest.fixture
    def stage(self, tmp_path):
        from historical_persona_pipeline.pipeline.conversion.conversion_stage import (
            ConversionStage,
        )

        return ConversionStage(load_config(), tmp_path)

    def _write_weights(self, directory, names):
        import torch
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {n: torch.zeros(2, 2) for n in names},
            str(directory / "model.safetensors"),
        )

    def test_modello_fuso_passa(self, stage, tmp_path):
        out = tmp_path / "merged"
        self._write_weights(out, [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ])
        stage._verify_merged(out)  # nessuna eccezione

    def test_struttura_peft_residua_solleva(self, stage, tmp_path):
        """Il caso reale: la conversione GGUF sarebbe fallita molto dopo."""
        out = tmp_path / "non_fuso"
        self._write_weights(out, [
            "model.layers.0.self_attn.q_proj.base_layer.weight",
            "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        ])
        with pytest.raises(RuntimeError, match="non ha incorporato l'adapter"):
            stage._verify_merged(out)

    def test_cartella_senza_pesi_non_solleva(self, stage, tmp_path):
        out = tmp_path / "vuota"
        out.mkdir()
        stage._verify_merged(out)  # nulla da verificare


class TestLlamaCppDiscovery:
    def test_build_di_unsloth_fra_i_percorsi_cercati(self, tmp_path):
        """Chi ha Unsloth Studio ha gia' una build di llama.cpp: l'export
        GGUF deve funzionare senza installare altro."""
        import inspect

        from historical_persona_pipeline.pipeline.conversion.conversion_stage import (
            ConversionStage,
        )

        source = inspect.getsource(ConversionStage._llama_cpp_roots)
        assert ".unsloth" in source
