from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass

import torch


@dataclass
class PretrainedMeasurement:
    regime: str
    score: str
    trainable_values: str
    train_step_seconds: str
    memory: str
    saved_weight: str
    note: str


def measure_pretrained_regimes(enabled: bool) -> list[PretrainedMeasurement]:
    if not enabled:
        return [
            PretrainedMeasurement(
                "extracteur gelé",
                "à mesurer avec --with-pretrained",
                "tête seule",
                "à mesurer",
                "à mesurer",
                "tête de classification",
                "téléchargement désactivé pendant le run standard",
            ),
            PretrainedMeasurement(
                "fine-tuning partiel",
                "à mesurer avec --with-pretrained",
                "dernières couches + tête",
                "à mesurer",
                "à mesurer",
                "couches modifiées + tête",
                "les couches basses restent plus stables que la sortie",
            ),
            PretrainedMeasurement(
                "adaptateurs",
                "à mesurer avec --with-pretrained",
                "petites matrices ajoutées",
                "à mesurer",
                "à mesurer",
                "adaptateurs + tête",
                "objectif : approcher le fine-tuning sans modifier le modèle emprunté",
            ),
        ]

    if importlib.util.find_spec("transformers") is None:
        return [
            PretrainedMeasurement(
                "modèle emprunté",
                "non mesuré",
                "non mesuré",
                "non mesuré",
                "non mesuré",
                "non mesuré",
                "installer transformers puis relancer --with-pretrained",
            )
        ]

    from transformers import AutoModel, AutoTokenizer

    model_name = "prajjwal1/bert-tiny"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    encoded = tokenizer(
        ["bright object moved silently across sky", "orange light hovered over trees"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**encoded).last_hidden_state[:, 0, :]
    elapsed = time.perf_counter() - started
    values = sum(parameter.numel() for parameter in model.parameters())
    frozen_trainable = output.shape[-1] * 18 + 18
    partial_trainable = sum(parameter.numel() for name, parameter in model.named_parameters() if "encoder.layer.1" in name)
    adapter_estimate = values // 100

    return [
        PretrainedMeasurement(
            "extracteur gelé",
            "à évaluer sur split phase 8",
            str(frozen_trainable),
            f"{elapsed:.3f}",
            "CPU, pic non tracé",
            f"{frozen_trainable * 4 / 1024:.1f} KiB",
            f"{model_name}, sortie CLS de taille {output.shape[-1]}",
        ),
        PretrainedMeasurement(
            "fine-tuning partiel",
            "à évaluer sur split phase 8",
            str(partial_trainable + frozen_trainable),
            f"{elapsed:.3f}",
            "CPU, pic non tracé",
            f"{(partial_trainable + frozen_trainable) * 4 / 1024 / 1024:.1f} MiB",
            "dernière couche encodeur + tête",
        ),
        PretrainedMeasurement(
            "adaptateurs",
            "à évaluer sur split phase 8",
            str(adapter_estimate + frozen_trainable),
            f"{elapsed:.3f}",
            "CPU, pic non tracé",
            f"{(adapter_estimate + frozen_trainable) * 4 / 1024 / 1024:.1f} MiB",
            "estimation LoRA/adaptateurs autour de 1 % du modèle",
        ),
    ]
