"""
Qwen-based AI detector with LoRA support.

Uses Qwen2.5-7B with optional LoRA adapters for AI text detection via prompting.
"""

from __future__ import annotations

import re
import torch
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger

from .base import (
    AIDetector,
    DetectorMetadata,
    DetectionChunk,
    DetectionFinal,
    DetectionResult,
    SimpleDetectionResult,
)


@dataclass
class QwenLoRAConfig:
    model_id: str
    lora_adapter_id: Optional[str]  # HuggingFace LoRA adapter ID or local path
    device: str
    size_gb: float
    languages: list[str]
    recommended: bool = False
    accuracy: float | None = None
    description: str | None = None


# Detection prompt template
DETECTION_PROMPT = """Analyze the following text and determine if it was written by an AI or a human.

Text to analyze:
\"\"\"
{text}
\"\"\"

Analyze the text for these AI-writing indicators:
1. Repetitive sentence structures
2. Overly formal or generic phrasing
3. Lack of personal voice or unique perspective
4. Perfect grammar with no natural variations
5. Predictable transitions and conclusions

Based on your analysis, respond with ONLY a JSON object in this exact format:
{{"is_ai": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""


class QwenLoRADetector(AIDetector):
    """Qwen-based AI detector using generative prompting with optional LoRA adapters."""

    def __init__(self, name: str, cache_dir: str, cfg: QwenLoRAConfig) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._actual_device = cfg.device  # May change at load time based on available VRAM
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
            logger.info(f"Loading Qwen LoRA detector: {self.metadata.model_id}")

            # Configure quantization based on AVAILABLE VRAM (not total)
            quantization_config = None
            use_cpu = False
            target_device = self.metadata.device
            
            if self.metadata.device == "cuda" and torch.cuda.is_available():
                # Check AVAILABLE memory, not total
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_vram_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
                
                logger.info(f"GPU VRAM: {free_vram_gb:.1f}GB free / {total_vram_gb:.1f}GB total")
                
                if free_vram_gb >= 12:
                    # Plenty of free VRAM - use FP16
                    quantization_config = None
                    logger.info(f"Using FP16 for Qwen detector ({free_vram_gb:.1f}GB free)")
                elif free_vram_gb >= 6:
                    # Limited VRAM - use 4-bit quantization
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info(f"Using 4-bit quantization for Qwen detector ({free_vram_gb:.1f}GB free)")
                else:
                    # Not enough VRAM - fall back to CPU
                    use_cpu = True
                    target_device = "cpu"
                    logger.warning(f"Only {free_vram_gb:.1f}GB VRAM free - using CPU for Qwen detector")
            else:
                use_cpu = True
                target_device = "cpu"

            # Load tokenizer (try offline first)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.metadata.model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                logger.info("Loaded tokenizer from local cache")
            except Exception:
                logger.info("Local tokenizer not found, downloading...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.metadata.model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )

            # Model loading kwargs
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if target_device == "cuda" else torch.float32,
            }

            if use_cpu:
                # CPU mode - no quantization, no device_map
                model_kwargs["device_map"] = None
                model_kwargs["low_cpu_mem_usage"] = True
            elif quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "auto"

            # Load base model (try offline first)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.metadata.model_id,
                    local_files_only=True,
                    **model_kwargs
                )
                logger.info("Loaded model from local cache")
            except Exception:
                logger.info("Local model not found, downloading...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.metadata.model_id,
                    **model_kwargs
                )

            # Load LoRA adapter if specified
            if self.cfg.lora_adapter_id:
                try:
                    from peft import PeftModel
                    logger.info(f"Loading LoRA adapter: {self.cfg.lora_adapter_id}")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        self.cfg.lora_adapter_id,
                        cache_dir=self.cache_dir,
                    )
                    logger.info("LoRA adapter loaded successfully")
                except ImportError:
                    logger.warning("PEFT not installed, skipping LoRA adapter loading")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA adapter: {e}")

            # Move model to CPU if needed (when not using device_map)
            if use_cpu and hasattr(self.model, 'to'):
                self.model = self.model.to("cpu")
                logger.info("Moved Qwen detector model to CPU")

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Store actual device for inference
            self._actual_device = target_device
            
            self._loaded = True
            logger.info(f"Loaded Qwen LoRA detector: {self.metadata.model_id} on {target_device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Qwen LoRA detector {self.metadata.model_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            logger.info(f"Unloaded Qwen LoRA detector: {self.metadata.model_id}")

    async def get_model_info(self):
        """Get model information."""
        return {
            "model_id": self.metadata.model_id,
            "loaded": self._loaded,
            "device": self.metadata.device,
            "lora_adapter": self.cfg.lora_adapter_id,
        }

    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse the model's JSON response."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                import json
                data = json.loads(json_match.group())
                is_ai = data.get("is_ai", False)
                confidence = float(data.get("confidence", 0.5))
                reason = data.get("reason", "")
                return is_ai, confidence, reason
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: look for keywords in response
        response_lower = response.lower()
        if "ai" in response_lower and ("written" in response_lower or "generated" in response_lower):
            if "not ai" in response_lower or "human" in response_lower:
                return False, 0.6, "Detected as human-written based on analysis"
            return True, 0.7, "Detected as AI-generated based on analysis"

        # Default to uncertain
        return False, 0.5, "Unable to determine with confidence"

    async def detect_async(self, text: str) -> SimpleDetectionResult:
        """Simple async detection using generative prompting."""
        if not self._loaded:
            await self.load()
        if not self._loaded:
            return SimpleDetectionResult(
                is_ai_generated=False,
                confidence=0.0,
                ai_probability=0.0,
                model_name=self.metadata.model_id,
                method="qwen_lora_error",
            )

        try:
            # Truncate text if too long (keep first 1500 chars for analysis)
            truncated_text = text[:1500] if len(text) > 1500 else text

            # Create the detection prompt
            prompt = DETECTION_PROMPT.format(text=truncated_text)

            # Tokenize with chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )

            # Move to model device (use stored actual device or detect from model)
            if hasattr(self, '_actual_device') and self._actual_device == "cpu":
                model_device = "cpu"
            else:
                model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,  # Low temperature for more deterministic output
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response (only the new tokens)
            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            )

            # Parse the response
            is_ai, confidence, reason = self._parse_response(response)

            logger.info(f"Qwen detector result: is_ai={is_ai}, confidence={confidence}, reason={reason[:100]}")

            return SimpleDetectionResult(
                is_ai_generated=is_ai,
                confidence=confidence,
                ai_probability=confidence if is_ai else 1 - confidence,
                model_name=self.metadata.model_id,
                method="qwen_lora_generative",
            )

        except Exception as e:
            logger.error(f"Error in Qwen LoRA detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return SimpleDetectionResult(
                is_ai_generated=False,
                confidence=0.0,
                ai_probability=0.0,
                model_name=self.metadata.model_id,
                method="qwen_lora_error",
            )

    async def detect(
        self, text: str, return_probabilities: bool = False
    ) -> AsyncGenerator[DetectionResult, None]:
        """Full detection with streaming support."""
        result = await self.detect_async(text)

        yield DetectionFinal(
            is_ai_generated=result.is_ai_generated,
            confidence=result.confidence,
            ai_probability=result.ai_probability,
            human_probability=1.0 - result.ai_probability,
            model_name=result.model_name,
            text_length=len(text),
            chunks_processed=1,
            method=result.method,
        )

