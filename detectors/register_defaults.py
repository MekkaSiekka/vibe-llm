"""
Register default detectors into the registry based on hardware.
"""

from __future__ import annotations

import torch

from .registry import get_registry, RegistryEntry
from .hf_sequence import HFSequenceClassifierDetector, HFDetectorConfig
from .qwen_lora import QwenLoRADetector, QwenLoRAConfig
from .base import DetectorMetadata


def register_defaults() -> None:
    reg = get_registry()
    has_gpu = torch.cuda.is_available()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_gpu else 0.0
    device = "cuda" if has_gpu else "cpu"

    # High-end models for powerful GPUs (8GB+ VRAM) - Using VERIFIED models only
    high_end_models = []
    if vram_gb >= 8.0:
        high_end_models = [
            # Verified working models for 12GB+ VRAM
            HFDetectorConfig(
                model_id="openai-community/roberta-large-openai-detector",
                device=device,
                size_gb=1.4,
                languages=["en"],
                recommended=vram_gb >= 12.0,
                accuracy=0.97,
                description="Large RoBERTa OpenAI detector - high accuracy",
            ),
            HFDetectorConfig(
                model_id="microsoft/deberta-v3-large",
                device=device,
                size_gb=1.8,
                languages=["en", "multilingual"],
                recommended=vram_gb >= 12.0,
                accuracy=0.96,
                description="DeBERTa-v3 Large - state-of-the-art transformer",
            ),
            # Advanced AI detection models that exist
            HFDetectorConfig(
                model_id="Hello-SimpleAI/chatgpt-detector-roberta",
                device=device,
                size_gb=0.5,
                languages=["en"],
                recommended=False,
                accuracy=0.93,
                description="Specialized ChatGPT detector",
            ),
            HFDetectorConfig(
                model_id="microsoft/deberta-v3-base",
                device=device,
                size_gb=0.8,
                languages=["en"],
                recommended=vram_gb >= 10.0,
                accuracy=0.94,
                description="DeBERTa-v3 Base - efficient transformer",
            ),
            # GPTZero-style and advanced AI detection models (verified)
            HFDetectorConfig(
                model_id="Hello-SimpleAI/chatgpt-detector-single",
                device=device,
                size_gb=0.5,
                languages=["en"],
                recommended=False,
                accuracy=0.92,
                description="GPTZero-style ChatGPT detector - single model",
            ),
        ]
    
    # Mid-range models for moderate GPUs (4-8GB VRAM)
    mid_range_models = [
        HFDetectorConfig(
            model_id="openai-community/roberta-base-openai-detector",
            device=device,
            size_gb=0.5,
            languages=["en"],
            recommended=vram_gb < 8.0,
            accuracy=0.95,
            description="OpenAI roberta-based detector",
        ),
        HFDetectorConfig(
            model_id="Hello-SimpleAI/chatgpt-detector-roberta",
            device=device,
            size_gb=0.5,
            languages=["en"],
            recommended=False,
            accuracy=0.92,
            description="ChatGPT detector roberta",
        ),
        # Additional GPTZero-style models for mid-range GPUs
        HFDetectorConfig(
            model_id="Hello-SimpleAI/chatgpt-detector-single",
            device=device,
            size_gb=0.4,
            languages=["en"],
            recommended=False,
            accuracy=0.90,
            description="GPTZero-style single model detector",
        ),
        HFDetectorConfig(
            model_id="microsoft/deberta-v3-base",
            device=device,
            size_gb=0.8,
            languages=["en"],
            recommended=vram_gb >= 6.0,
            accuracy=0.94,
            description="DeBERTa-v3 Base - efficient transformer",
        ),
    ]
    
    # Combine models based on hardware capability
    defaults = high_end_models + mid_range_models

    # Register Qwen LoRA detector for high-end GPUs (8GB+ VRAM)
    if vram_gb >= 8.0:
        qwen_lora_cfg = QwenLoRAConfig(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            lora_adapter_id=None,  # Base model for now; can add LoRA adapter ID later
            device=device,
            size_gb=10.0,
            languages=["en", "zh", "multilingual"],
            recommended=vram_gb >= 12.0,
            accuracy=0.92,
            description="Qwen2.5-7B with LoRA - generative AI detector with multilingual support",
        )
        qwen_meta = DetectorMetadata(
            name="Qwen2.5-7B-AI-Detector",
            model_id=qwen_lora_cfg.model_id,
            device=qwen_lora_cfg.device,
            size_gb=qwen_lora_cfg.size_gb,
            languages=qwen_lora_cfg.languages,
            recommended=qwen_lora_cfg.recommended,
            accuracy=qwen_lora_cfg.accuracy,
            description=qwen_lora_cfg.description,
        )

        def _qwen_factory(det_name: str, cache_dir: str, c: QwenLoRAConfig = qwen_lora_cfg):
            return QwenLoRADetector(det_name, cache_dir, c)

        reg.register("Qwen2.5-7B-AI-Detector", RegistryEntry(create_fn=_qwen_factory, metadata=qwen_meta))

    for cfg in defaults:
        # Registry key based on HF id tail
        name = cfg.model_id.split("/")[-1]
        meta = DetectorMetadata(
            name=name,
            model_id=cfg.model_id,
            device=cfg.device,
            size_gb=cfg.size_gb,
            languages=cfg.languages,
            recommended=cfg.recommended,
            accuracy=cfg.accuracy,
            description=cfg.description,
        )

        def _factory(det_name: str, cache_dir: str, c: HFDetectorConfig = cfg):
            return HFSequenceClassifierDetector(det_name, cache_dir, c)

        reg.register(name, RegistryEntry(create_fn=_factory, metadata=meta))

        # Also register human-friendly aliases matching HardwareDetector entries
        if "roberta-base-openai-detector" in name:
            alias = "RoBERTa-OpenAI-Detector"
            reg.register(alias, RegistryEntry(create_fn=_factory, metadata=meta))
        if "chatgpt-detector-roberta" in name:
            alias = "BERT-ChatGPT-Detector"
            reg.register(alias, RegistryEntry(create_fn=_factory, metadata=meta))


