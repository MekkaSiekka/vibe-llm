"""
Detectors package

Provides base interfaces, concrete implementations, and a registry/factory
for AI text detection models.
"""

from .base import (
    DetectorMetadata,
    DetectionChunk,
    DetectionFinal,
    DetectionResult,
    AIDetector,
)
from .registry import get_registry, create_detector

__all__ = [
    "DetectorMetadata",
    "DetectionChunk",
    "DetectionFinal",
    "DetectionResult",
    "AIDetector",
    "get_registry",
    "create_detector",
]


