"""
Detector registry and factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .base import AIDetector, DetectorMetadata


@dataclass
class RegistryEntry:
    create_fn: Callable[[str, str], AIDetector]
    metadata: DetectorMetadata


class DetectorRegistry:
    def __init__(self) -> None:
        self._name_to_entry: Dict[str, RegistryEntry] = {}

    def register(self, name: str, entry: RegistryEntry) -> None:
        self._name_to_entry[name] = entry

    def get(self, name: str) -> Optional[RegistryEntry]:
        return self._name_to_entry.get(name)

    def list(self) -> Dict[str, DetectorMetadata]:
        return {name: entry.metadata for name, entry in self._name_to_entry.items()}


_REGISTRY: Optional[DetectorRegistry] = None


def get_registry() -> DetectorRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DetectorRegistry()
    return _REGISTRY


def create_detector(name: str, cache_dir: str) -> AIDetector:
    reg = get_registry()
    entry = reg.get(name)
    if not entry:
        raise ValueError(f"Detector '{name}' not registered")
    return entry.create_fn(name, cache_dir)


