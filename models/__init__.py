"""
Local LLM Model Management System

Provides hot-switchable model management with hardware detection and compatibility checking.
"""

from .manager import ModelManager
from .detector import HardwareDetector
from .qwen import QwenModel

__all__ = ["ModelManager", "HardwareDetector", "QwenModel"]

