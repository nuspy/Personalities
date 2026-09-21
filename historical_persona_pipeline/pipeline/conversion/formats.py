"""Formati di esportazione e livelli di quantizzazione.

Un adapter LoRA da solo non si usa: pesa pochi MB ma richiede il modello base
a fianco e una libreria che sappia applicarlo. Per portare un personaggio
altrove — Ollama, LM Studio, llama.cpp, un altro computer — serve convertirlo.
Qui sono descritti i formati di destinazione e i compromessi di ciascuno.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ExportFormat(str, Enum):
    MERGED_16BIT = "merged_16bit"
    MERGED_4BIT = "merged_4bit"
    ADAPTER_ONLY = "adapter_only"
    GGUF = "gguf"
    OLLAMA = "ollama"

    @property
    def label(self) -> str:
        return _FORMAT_LABELS[self]

    @property
    def description(self) -> str:
        return _FORMAT_DESCRIPTIONS[self]

    @property
    def needs_gguf_toolchain(self) -> bool:
        return self in (ExportFormat.GGUF, ExportFormat.OLLAMA)


_FORMAT_LABELS: Dict[ExportFormat, str] = {
    ExportFormat.MERGED_16BIT: "Modello completo (16 bit)",
    ExportFormat.MERGED_4BIT: "Modello completo (4 bit)",
    ExportFormat.ADAPTER_ONLY: "Solo adapter LoRA",
    ExportFormat.GGUF: "GGUF (llama.cpp, LM Studio)",
    ExportFormat.OLLAMA: "Ollama (GGUF + Modelfile)",
}

_FORMAT_DESCRIPTIONS: Dict[ExportFormat, str] = {
    ExportFormat.MERGED_16BIT: (
        "L'adapter viene fuso nei pesi del modello, che diventa autonomo. "
        "Massima qualita' e punto di partenza per ogni altra conversione. "
        "Occupa circa 2 GB per miliardo di parametri (un 8B sono ~16 GB)."
    ),
    ExportFormat.MERGED_4BIT: (
        "Come il precedente ma quantizzato a 4 bit: circa un quarto dello "
        "spazio, con una perdita di qualita' contenuta. Utile per distribuire "
        "il modello a chi ha poca VRAM."
    ),
    ExportFormat.ADAPTER_ONLY: (
        "Copia del solo adapter (pochi MB), con i metadati di provenienza. "
        "Richiede il modello base a fianco, ma e' il formato piu' leggero da "
        "archiviare e da combinare con altri adapter."
    ),
    ExportFormat.GGUF: (
        "Formato di llama.cpp: un unico file eseguibile su CPU, GPU o Apple "
        "Silicon. E' quello che caricano LM Studio, Jan e la maggior parte "
        "dei runner locali."
    ),
    ExportFormat.OLLAMA: (
        "GGUF piu' un Modelfile con il prompt di sistema del personaggio gia' "
        "incorporato: dopo l'importazione si conversa col personaggio senza "
        "doverlo configurare."
    ),
}


@dataclass(frozen=True)
class Quantization:
    name: str
    description: str
    bits_per_weight: float

    def estimated_size_gb(self, parameters_billions: float) -> float:
        return parameters_billions * self.bits_per_weight / 8


# Livelli GGUF usati nella pratica; l'elenco completo di llama.cpp e' molto
# piu' lungo ma il resto o e' obsoleto o non vale il compromesso.
QUANTIZATIONS: Dict[str, Quantization] = {
    "f16": Quantization(
        "f16", "Nessuna quantizzazione: qualita' piena, dimensione massima.", 16.0
    ),
    "q8_0": Quantization(
        "q8_0", "8 bit: differenza dal modello pieno praticamente impercettibile.", 8.5
    ),
    "q6_k": Quantization(
        "q6_k", "6 bit: perdita minima, buon compromesso se la memoria basta.", 6.6
    ),
    "q5_k_m": Quantization(
        "q5_k_m", "5 bit: perdita contenuta, dimensione ragionevole.", 5.7
    ),
    "q4_k_m": Quantization(
        "q4_k_m",
        "4 bit: il compromesso piu' usato. Qualita' buona, gira su hardware modesto.",
        4.8,
    ),
    "q3_k_m": Quantization(
        "q3_k_m", "3 bit: degrado percepibile, da usare solo se la memoria e' poca.", 3.9
    ),
    "q2_k": Quantization(
        "q2_k", "2 bit: degrado marcato. Utile solo per provare che il modello gira.", 3.0
    ),
}

DEFAULT_QUANTIZATION = "q4_k_m"

# Ordine di presentazione: dalla qualita' piena alla piu' compressa.
QUANTIZATION_ORDER: List[str] = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]


def estimate_parameters_billions(model_name: str) -> float:
    """Dimensione del modello dedotta dal nome, per stimare lo spazio su disco.

    Non e' esatta — serve solo ad avvisare prima di una conversione che
    occuperebbe decine di GB.
    """
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 8.0  # ipotesi prudente: la taglia piu' diffusa
