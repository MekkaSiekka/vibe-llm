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
            # High-end GPU models - start with smaller models for faster testing
            compatible_models.extend([
                {
                    "name": "Qwen3-0.6B",
                    "model_id": "Qwen/Qwen3-0.6B",
                    "size_gb": 1.2,
                    "languages": ["en", "zh"],
                    "recommended": True,
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

