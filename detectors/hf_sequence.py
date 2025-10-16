"""
HuggingFace sequence classification detector implementation.
"""

from __future__ import annotations

from typing import AsyncGenerator
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from loguru import logger

from .base import AIDetector, DetectorMetadata, DetectionChunk, DetectionFinal, DetectionResult


@dataclass
class HFDetectorConfig:
    model_id: str
    device: str
    size_gb: float
    languages: list[str]
    recommended: bool = False
    accuracy: float | None = None
    description: str | None = None


class HFSequenceClassifierDetector(AIDetector):
    def __init__(self, name: str, cache_dir: str, cfg: HFDetectorConfig) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.detection_threshold = 0.5
        self.metadata = DetectorMetadata(
            name=name,
            model_id=cfg.model_id,
            device=cfg.device,
            size_gb=cfg.size_gb,
            languages=cfg.languages,
            recommended=cfg.recommended,
            accuracy=cfg.accuracy,
            description=cfg.description,
        )

    async def load(self) -> bool:
        if self._loaded:
            return True
        try:
            logger.info(f"Loading HF detector: {self.metadata.model_id}")
            quantization_config = None
            if self.metadata.device == "cuda" and torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 8:
                    quantization_config = None
                elif vram_gb >= 4:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.metadata.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.metadata.device == "cuda" else torch.float32,
            }
            if quantization_config:
                kwargs["quantization_config"] = quantization_config
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = "auto" if self.metadata.device == "cuda" else None

            self.model = AutoModelForSequenceClassification.from_pretrained(self.metadata.model_id, **kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self._loaded = True
            logger.info(f"Loaded HF detector: {self.metadata.model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load detector {self.metadata.model_id}: {e}")
            return False

    async def unload(self) -> None:
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def detect_async(self, text: str) -> "SimpleDetectionResult":
        """Simple async detection method for the new architecture."""
        if not self._loaded:
            await self.load()
        if not self._loaded:
            from .base import SimpleDetectionResult
            return SimpleDetectionResult(
                is_ai_generated=False,
                confidence=0.0,
                ai_probability=0.0,
                model_name=self.metadata.model_id,
                method="hf_sequence_classification"
            )
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            # Move inputs to the same device as the model
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            elif self.metadata.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get predictions
                ai_prob = probabilities[0][1].item()  # Assuming class 1 is AI
                human_prob = probabilities[0][0].item()  # Assuming class 0 is human
                is_ai = ai_prob > 0.5
                confidence = max(ai_prob, human_prob)
            
            from .base import SimpleDetectionResult
            return SimpleDetectionResult(
                is_ai_generated=is_ai,
                confidence=confidence,
                ai_probability=ai_prob,
                model_name=self.metadata.model_id,
                method="hf_sequence_classification"
            )
            
        except Exception as e:
            logger.error(f"Error in HF detection: {e}")
            from .base import SimpleDetectionResult
            return SimpleDetectionResult(
                is_ai_generated=False,
                confidence=0.0,
                ai_probability=0.0,
                model_name=self.metadata.model_id,
                method="hf_sequence_classification_error"
            )

    async def detect(self, text: str, return_probabilities: bool = False) -> AsyncGenerator[DetectionResult, None]:
        if not self._loaded:
            await self.load()
        if not self._loaded:
            yield DetectionFinal(
                is_ai_generated=False,
                confidence=0.0,
                ai_probability=0.0,
                human_probability=1.0,
                model_name=self.metadata.model_id,
                text_length=len(text),
                chunks_processed=0,
                method="hf_sequence_classifier",
            )
            return

        # Tokenize and run inference in a single chunk for now (can be extended)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        if self.metadata.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            if probs.shape[1] == 2:
                human_prob = probs[0][0].item()
                ai_prob = probs[0][1].item()
            else:
                ai_prob = probs[0][-1].item()
                human_prob = 1.0 - ai_prob

        is_ai = ai_prob > self.detection_threshold
        confidence = max(ai_prob, human_prob)

        # Final result only; streaming chunks are trivial to add here
        yield DetectionFinal(
            is_ai_generated=is_ai,
            confidence=confidence,
            ai_probability=ai_prob,
            human_probability=human_prob,
            model_name=self.metadata.model_id,
            text_length=len(text),
            chunks_processed=1,
            method="hf_sequence_classifier",
        )


