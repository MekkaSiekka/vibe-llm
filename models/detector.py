"""
Hardware Detection and Compatibility Module

Detects system capabilities and determines which models can run efficiently.
"""

import platform
import psutil
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class HardwareSpecs:
    """Hardware specifications for model compatibility checking."""
    cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    has_gpu: bool
    gpu_memory_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    platform: str = "unknown"
    architecture: str = "unknown"


class HardwareDetector:
    """Detects and analyzes system hardware capabilities."""
    
    def __init__(self):
        self.specs = self._detect_hardware()
        logger.info(f"Hardware detected: {self.specs}")
    
    def _detect_hardware(self) -> HardwareSpecs:
        """Detect current hardware specifications."""
        # CPU information
        cpu_cores = psutil.cpu_count(logical=False)
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        # Platform information
        platform_name = platform.system().lower()
        architecture = platform.machine().lower()
        
        # GPU detection
        has_gpu = torch.cuda.is_available()
        gpu_memory = None
        gpu_name = None
        
        if has_gpu:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return HardwareSpecs(
            cpu_cores=cpu_cores,
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            has_gpu=has_gpu,
            gpu_memory_gb=gpu_memory,
            gpu_name=gpu_name,
            platform=platform_name,
            architecture=architecture
        )
    
    def get_compatible_models(self) -> List[Dict[str, any]]:
        """Get list of models compatible with current hardware."""
        compatible_models = []
        
        # Qwen model variants based on hardware
        if self.specs.has_gpu and self.specs.gpu_memory_gb >= 8:
            # High-end GPU models - prioritize quality models for powerful hardware
            compatible_models.extend([
                {
                    "name": "Qwen2.5-7B-Instruct",
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "size_gb": 10,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": True,
                    "device": "cuda"
                },
                # 14B FP16 variant for GPUs with >=28GB VRAM. This uses more VRAM for higher quality.
                {
                    "name": "Qwen2.5-14B-Instruct-FP16",
                    "model_id": "Qwen/Qwen2.5-14B-Instruct",
                    "size_gb": 28,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": self.specs.gpu_memory_gb is not None and self.specs.gpu_memory_gb >= 28,
                    "device": "cuda",
                    "precision_mode": "fp16",
                    "description": "Forces FP16 loading (no quant) on >=28GB VRAM to use ~28-32GB"
                },
                # 14B fits well on 24GB+ VRAM with FP16 or on 16GB+ with 8-bit
                {
                    "name": "Qwen2.5-14B-Instruct",
                    "model_id": "Qwen/Qwen2.5-14B-Instruct",
                    "size_gb": 28,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": self.specs.gpu_memory_gb is not None and self.specs.gpu_memory_gb >= 24,
                    "device": "cuda"
                },
                # Very large model; requires quantization on single GPU. Offer when VRAM is huge
                {
                    "name": "Qwen2.5-32B-Instruct",
                    "model_id": "Qwen/Qwen2.5-32B-Instruct",
                    "size_gb": 64,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": self.specs.gpu_memory_gb is not None and self.specs.gpu_memory_gb >= 48,
                    "device": "cuda"
                },
                {
                    "name": "Qwen3-4B-Instruct",
                    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                    "size_gb": 8,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": False,
                    "device": "cuda"
                },
                {
                    "name": "Qwen2.5-3B-Instruct",
                    "model_id": "Qwen/Qwen2.5-3B-Instruct",
                    "size_gb": 6,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": False,
                    "device": "cuda"
                },
                {
                    "name": "Mistral-7B-Instruct",
                    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                    "size_gb": 13,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda"
                },
                {
                    "name": "Phi-3-mini-4k-instruct",
                    "model_id": "microsoft/Phi-3-mini-4k-instruct",
                    "size_gb": 3.8,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda"
                },
                {
                    "name": "DeepSeek-R1-Distill-Llama-8B",
                    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "size_gb": 15,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda"
                },
                {
                    "name": "Llama-3.1-8B-Instruct",
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "size_gb": 16,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda"
                },
                {
                    "name": "Gemma-2-9B-it",
                    "model_id": "google/gemma-2-9b-it",
                    "size_gb": 18,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda"
                },
                {
                    "name": "Qwen3-0.6B",
                    "model_id": "Qwen/Qwen3-0.6B",
                    "size_gb": 1.2,
                    "languages": ["en", "zh"],
                    "recommended": False,
                    "device": "cuda"
                }
            ])
        elif self.specs.has_gpu and self.specs.gpu_memory_gb >= 4:
            # Mid-range GPU models
            compatible_models.extend([
                {
                    "name": "Qwen2.5-3B-Instruct",
                    "model_id": "Qwen/Qwen2.5-3B-Instruct",
                    "size_gb": 6,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": True,
                    "device": "cuda"
                }
            ])
        
        # CPU-only models
        if self.specs.available_memory_gb >= 8:
            compatible_models.extend([
                {
                    "name": "Qwen3-4B-Instruct-CPU",
                    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                    "size_gb": 8,
                    "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
                    "recommended": False,
                    "device": "cpu"
                }
            ])
        
        # Mobile/Edge models for iOS/Android compatibility
        if self.specs.available_memory_gb >= 2:
            compatible_models.extend([
                {
                    "name": "Qwen3-0.6B",
                    "model_id": "Qwen/Qwen3-0.6B",
                    "size_gb": 1.2,
                    "languages": ["en", "zh"],
                    "recommended": False,
                    "device": "cpu",
                    "mobile_optimized": True
                }
            ])
        
        # Add AI Detection Models
        ai_detection_models = []
        
        # High-end AI detection models (8GB+ VRAM)
        if self.specs.has_gpu and self.specs.gpu_memory_gb >= 8:
            ai_detection_models.extend([
                {
                    "name": "roberta-large-openai-detector",
                    "model_id": "openai-community/roberta-large-openai-detector",
                    "size_gb": 1.4,
                    "languages": ["en"],
                    "model_type": "ai_detector",
                    "recommended": self.specs.gpu_memory_gb >= 12,
                    "device": "cuda",
                    "accuracy": 0.97,
                    "description": "Large RoBERTa OpenAI detector - high accuracy"
                },
                {
                    "name": "deberta-v3-large",
                    "model_id": "microsoft/deberta-v3-large",
                    "size_gb": 1.8,
                    "languages": ["en"],
                    "model_type": "ai_detector",
                    "recommended": self.specs.gpu_memory_gb >= 12,
                    "device": "cuda",
                    "accuracy": 0.96,
                    "description": "DeBERTa-v3 Large - state-of-the-art transformer"
                },
                {
                    "name": "chatgpt-detector-roberta",
                    "model_id": "Hello-SimpleAI/chatgpt-detector-roberta",
                    "size_gb": 0.5,
                    "languages": ["en"],
                    "model_type": "ai_detector",
                    "recommended": False,
                    "device": "cuda",
                    "accuracy": 0.93,
                    "description": "Specialized ChatGPT detector"
                }
            ])
        
        # Mid-range AI detection models (4-8GB VRAM or CPU)
        if (self.specs.has_gpu and self.specs.gpu_memory_gb >= 4) or self.specs.available_memory_gb >= 4:
            device = "cuda" if self.specs.has_gpu else "cpu"
            ai_detection_models.extend([
                {
                    "name": "roberta-base-openai-detector",
                    "model_id": "openai-community/roberta-base-openai-detector",
                    "size_gb": 0.5,
                    "languages": ["en"],
                    "model_type": "ai_detector",
                    "recommended": self.specs.gpu_memory_gb < 8 if self.specs.has_gpu else True,
                    "device": device,
                    "accuracy": 0.95,
                    "description": "OpenAI roberta-based detector - efficient and accurate"
                },
                {
                    "name": "chatgpt-detector-single",
                    "model_id": "Hello-SimpleAI/chatgpt-detector-single",
                    "size_gb": 0.4,
                    "languages": ["en"],
                    "model_type": "ai_detector",
                    "recommended": False,
                    "device": device,
                    "accuracy": 0.90,
                    "description": "GPTZero-style single model detector"
                },
                {
                    "name": "deberta-v3-base",
                    "model_id": "microsoft/deberta-v3-base",
                    "size_gb": 0.8,
                    "languages": ["en"],
                    "model_type": "ai_detector",
                    "recommended": False,
                    "device": device,
                    "accuracy": 0.94,
                    "description": "DeBERTa-v3 Base - efficient transformer"
                }
            ])
        
        compatible_models.extend(ai_detection_models)
        
        return compatible_models
    
    def estimate_performance(self, model_size_gb: float) -> Dict[str, any]:
        """Estimate model performance based on hardware specs."""
        if self.specs.has_gpu:
            # GPU performance estimation
            if model_size_gb <= self.specs.gpu_memory_gb * 0.8:
                return {
                    "device": "cuda",
                    "estimated_tokens_per_second": 50,
                    "memory_efficient": True,
                    "recommended": True
                }
            else:
                return {
                    "device": "cpu",
                    "estimated_tokens_per_second": 5,
                    "memory_efficient": False,
                    "recommended": False
                }
        else:
            # CPU performance estimation
            if model_size_gb <= self.specs.available_memory_gb * 0.7:
                return {
                    "device": "cpu",
                    "estimated_tokens_per_second": 3,
                    "memory_efficient": True,
                    "recommended": True
                }
            else:
                return {
                    "device": "cpu",
                    "estimated_tokens_per_second": 1,
                    "memory_efficient": False,
                    "recommended": False
                }
    
    def get_system_info(self) -> Dict[str, any]:
        """Get comprehensive system information."""
        return {
            "hardware": {
                "cpu_cores": self.specs.cpu_cores,
                "total_memory_gb": round(self.specs.total_memory_gb, 2),
                "available_memory_gb": round(self.specs.available_memory_gb, 2),
                "has_gpu": self.specs.has_gpu,
                "gpu_memory_gb": round(self.specs.gpu_memory_gb, 2) if self.specs.gpu_memory_gb else None,
                "gpu_name": self.specs.gpu_name,
                "platform": self.specs.platform,
                "architecture": self.specs.architecture
            },
            "compatible_models": self.get_compatible_models(),
            "recommendations": {
                "best_model": next((m for m in self.get_compatible_models() if m.get("recommended")), None),
                "mobile_optimized": [m for m in self.get_compatible_models() if m.get("mobile_optimized")]
            }
        }

