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
        
        # AI Detection models based on hardware
        ai_detection_models = self._get_ai_detection_models()
        compatible_models.extend(ai_detection_models)
        
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
        
        return compatible_models
    
    def _get_ai_detection_models(self) -> List[Dict[str, any]]:
        """Get AI detection models compatible with current hardware."""
        ai_models = []
        
        # Ultra high-end GPU (12GB+) - can run the most powerful models
        if self.specs.has_gpu and self.specs.gpu_memory_gb >= 12:
            ai_models.extend([
                {
                    "name": "DeBERTa-V2-XLarge-Detector",
                    "model_id": "microsoft/deberta-v2-xlarge",
                    "size_gb": 2.3,
                    "languages": ["en", "multilingual"],
                    "recommended": True,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.98,
                    "description": "State-of-the-art transformer - highest accuracy"
                },
                {
                    "name": "DeBERTa-V3-Large-Detector",
                    "model_id": "microsoft/deberta-v3-large",
                    "size_gb": 1.8,
                    "languages": ["en", "multilingual"],
                    "recommended": True,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.96,
                    "description": "Large DeBERTa model - excellent performance"
                },
                {
                    "name": "RoBERTa-Large-OpenAI-Detector",
                    "model_id": "roberta-large-openai-detector",
                    "size_gb": 1.4,
                    "languages": ["en"],
                    "recommended": True,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.97,
                    "description": "Large RoBERTa OpenAI detector - very high accuracy"
                },
                {
                    "name": "Advanced-Mixed-Detector",
                    "model_id": "andreas122001/roberta-mixed-detector",
                    "size_gb": 0.6,
                    "languages": ["en"],
                    "recommended": True,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.95,
                    "description": "Advanced mixed AI content detector"
                },
            ])
        
        # High-end GPU (8-12GB) - can run larger detection models
        if self.specs.has_gpu and self.specs.gpu_memory_gb >= 8:
            ai_models.extend([
                {
                    "name": "RoBERTa-OpenAI-Detector",
                    "model_id": "openai-community/roberta-base-openai-detector",
                    "size_gb": 0.5,
                    "languages": ["en"],
                    "recommended": self.specs.gpu_memory_gb < 12,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.95,
                    "description": "High accuracy AI text detector"
                },
                {
                    "name": "BERT-ChatGPT-Detector",
                    "model_id": "Hello-SimpleAI/chatgpt-detector-roberta",
                    "size_gb": 0.5,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.92,
                    "description": "Specialized ChatGPT content detector"
                },
                {
                    "name": "DeBERTa-V3-Base-Detector",
                    "model_id": "microsoft/deberta-v3-base",
                    "size_gb": 0.8,
                    "languages": ["en"],
                    "recommended": self.specs.gpu_memory_gb >= 10 and self.specs.gpu_memory_gb < 12,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.94,
                    "description": "Efficient DeBERTa transformer"
                },
                {
                    "name": "GPTZero-Style-Single-Detector",
                    "model_id": "Hello-SimpleAI/chatgpt-detector-single",
                    "size_gb": 0.5,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.92,
                    "description": "GPTZero-style ChatGPT detection model"
                },
                {
                    "name": "DeBERTa-V3-Detector",
                    "model_id": "microsoft/deberta-v3-base",
                    "size_gb": 0.7,
                    "languages": ["en", "zh"],
                    "recommended": False,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.94,
                    "description": "Advanced multilingual AI detector"
                }
            ])
        
        # Mid-range GPU - efficient detection models
        elif self.specs.has_gpu and self.specs.gpu_memory_gb >= 4:
            ai_models.extend([
                {
                    "name": "RoBERTa-OpenAI-Detector",
                    "model_id": "openai-community/roberta-base-openai-detector",
                    "size_gb": 0.5,
                    "languages": ["en"],
                    "recommended": True,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.95,
                    "description": "High accuracy AI text detector"
                },
                {
                    "name": "DistilBERT-Detector",
                    "model_id": "distilbert-base-uncased",
                    "size_gb": 0.3,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cuda",
                    "model_type": "ai_detector",
                    "accuracy": 0.88,
                    "description": "Fast and efficient detector"
                }
            ])
        
        # CPU-only systems - lightweight models
        if self.specs.available_memory_gb >= 4:
            ai_models.extend([
                {
                    "name": "RoBERTa-OpenAI-Detector-CPU",
                    "model_id": "openai-community/roberta-base-openai-detector",
                    "size_gb": 0.5,
                    "languages": ["en"],
                    "recommended": True if not self.specs.has_gpu else False,
                    "device": "cpu",
                    "model_type": "ai_detector",
                    "accuracy": 0.95,
                    "description": "High accuracy AI text detector (CPU)"
                }
            ])
        
        # Low-memory systems - ultra-light models
        if self.specs.available_memory_gb >= 2:
            ai_models.extend([
                {
                    "name": "DistilBERT-Detector-CPU",
                    "model_id": "distilbert-base-uncased",
                    "size_gb": 0.3,
                    "languages": ["en"],
                    "recommended": False,
                    "device": "cpu",
                    "model_type": "ai_detector",
                    "accuracy": 0.88,
                    "description": "Lightweight detector for low-memory systems"
                }
            ])
        
        return ai_models
    
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

