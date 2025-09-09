"""
Local LLM Service

Provides REST and WebSocket APIs for local LLM interactions.
"""

from .main import app, model_manager

__all__ = ["app", "model_manager"]

