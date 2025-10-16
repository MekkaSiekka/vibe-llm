"""
Detector base interfaces and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Protocol, runtime_checkable, Literal, Union


@dataclass(frozen=True)
class DetectorMetadata:
    name: str
    model_id: str
    device: Literal["cpu", "cuda", "auto"]
    size_gb: float
    languages: List[str]
    recommended: bool = False
    accuracy: Optional[float] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class DetectionChunk:
    chunk_id: int
    is_ai_generated: bool
    ai_probability: float
    human_probability: float
    text_preview: str


@dataclass(frozen=True)
class DetectionFinal:
    is_ai_generated: bool
    confidence: float
    ai_probability: float
    human_probability: float
    model_name: str
    text_length: int
    chunks_processed: int
    method: str


DetectionResult = Union[DetectionChunk, DetectionFinal]


@dataclass
class SimpleDetectionResult:
    """Simple detection result for the new architecture."""
    is_ai_generated: bool
    confidence: float
    ai_probability: float
    model_name: str
    method: str


@runtime_checkable
class AIDetector(Protocol):
    """Protocol that all detectors must implement."""

    metadata: DetectorMetadata

    async def load(self) -> bool:
        ...

    async def unload(self) -> None:
        ...

    async def detect(
        self,
        text: str,
        return_probabilities: bool = False,
    ) -> AsyncGenerator[DetectionResult, None]:
        ...
    
    async def detect_async(self, text: str) -> SimpleDetectionResult:
        """Simple async detection method for the new architecture."""
        ...


