"""
Model Manager

Centralized model management with hot-switching, availability checking, and hardware optimization.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
import json
from pathlib import Path

from .detector import HardwareDetector
from .qwen import QwenModel
from .exllama import ExLlamaModel
from detectors.registry import get_registry
from detectors.base import AIDetector
from detectors.register_defaults import register_defaults


@dataclass
class ModelInfo:
    """Model information structure."""
    name: str
    model_id: str
    size_gb: float
    languages: List[str]
    device: str
    model_type: str = "chat"  # "chat" or "ai_detector"
    loaded: bool = False
    available: bool = False
    recommended: bool = False
    mobile_optimized: bool = False
    accuracy: Optional[float] = None  # For AI detectors
    description: Optional[str] = None  # Model description
    precision_mode: Optional[str] = None  # e.g., "fp16" to force full precision on big GPUs
    quantization_format: Optional[str] = None  # e.g., "exl2" for ExLlamaV2 models


class ModelManager:
    """Manages all available models with hot-switching capabilities."""
    
    def __init__(self, cache_dir: str = "./models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.hardware_detector = HardwareDetector()
        
        # Current models by type
        self.current_chat_model: Optional[QwenModel] = None
        self.current_ai_detector: Optional[AIDetector] = None
        
        # Detector registry for AI detection models
        self.detector_registry = get_registry()
        
        # Register default detectors
        register_defaults()
        
        # Backwards compatibility: maintain current_model as alias to current_chat_model
        self._current_model = None
        
        # Model storage
        self.available_models: Dict[str, ModelInfo] = {}
        self.model_instances: Dict[str, Any] = {}  # Can hold QwenModel or AIDetectorModel
        
        # Concurrency controls
        self._load_locks: Dict[str, asyncio.Lock] = {"chat": asyncio.Lock(), "ai_detector": asyncio.Lock()}
        self._generation_semaphore = asyncio.Semaphore(1)
        self._detection_semaphore = asyncio.Semaphore(1)
        
        # Initialize available models
        self._initialize_models()
    
    @property
    def current_model(self):
        """Backwards compatibility property for current model - returns any loaded model."""
        # For backwards compatibility, return any currently loaded model
        if self.current_chat_model:
            return self.current_chat_model
        elif self.current_ai_detector:
            return self.current_ai_detector
        return None
    
    @current_model.setter
    def current_model(self, value):
        """Backwards compatibility setter - sets the appropriate model type."""
        # Try to determine what kind of model this is
        # Chat models have model_id directly, detectors have metadata.model_id
        value_model_id = None
        if hasattr(value, 'model_id'):
            value_model_id = value.model_id
        elif hasattr(value, 'metadata') and hasattr(value.metadata, 'model_id'):
            value_model_id = value.metadata.model_id
            
        if value_model_id:
            # Look up model type in our available models
            for model_info in self.available_models.values():
                if model_info.model_id == value_model_id:
                    if model_info.model_type == "chat":
                        self.current_chat_model = value
                    elif model_info.model_type == "ai_detector":
                        self.current_ai_detector = value
                    return
        
        # Fallback: assume it's a chat model for backwards compatibility
        self.current_chat_model = value
    
    def _initialize_models(self):
        """Initialize available models based on hardware compatibility."""
        compatible_models = self.hardware_detector.get_compatible_models()
        
        for model_data in compatible_models:
            model_info = ModelInfo(
                name=model_data["name"],
                model_id=model_data["model_id"],
                size_gb=model_data["size_gb"],
                languages=model_data["languages"],
                device=model_data["device"],
                model_type=model_data.get("model_type", "chat"),  # Default to chat model
                recommended=model_data.get("recommended", False),
                mobile_optimized=model_data.get("mobile_optimized", False),
                accuracy=model_data.get("accuracy", None),
                description=model_data.get("description", None),
                precision_mode=model_data.get("precision_mode", None),
                quantization_format=model_data.get("quantization_format", None),
                available=self._check_model_availability(model_data["model_id"])
            )
            
            self.available_models[model_info.name] = model_info
            
            # Create appropriate model instance based on type
            if model_info.model_type == "ai_detector":
                # Use the new detector registry to create instances
                try:
                    from detectors.registry import create_detector
                    detector_instance = create_detector(model_info.name, str(self.cache_dir))
                    self.model_instances[model_info.name] = detector_instance
                except Exception as e:
                    logger.warning(f"Failed to create detector instance for {model_info.name}: {e}")
            elif model_info.quantization_format == "exl2":
                # Use ExLlamaModel for EXL2 quantized models
                self.model_instances[model_info.name] = ExLlamaModel(
                    model_id=model_info.model_id,
                    cache_dir=str(self.cache_dir),
                    device=model_info.device,
                    precision_mode=model_info.precision_mode
                )
            else:  # Default to QwenModel for standard HuggingFace models
                self.model_instances[model_info.name] = QwenModel(
                    model_id=model_info.model_id,
                    cache_dir=str(self.cache_dir),
                    device=model_info.device,
                    precision_mode=model_info.precision_mode
                )
        
        chat_models = len([m for m in self.available_models.values() if m.model_type == "chat"])
        exl2_models = len([m for m in self.available_models.values() if m.quantization_format == "exl2"])
        ai_detectors = len([m for m in self.available_models.values() if m.model_type == "ai_detector"])
        logger.info(f"Initialized {chat_models} chat models ({exl2_models} EXL2) and {ai_detectors} AI detection models")
    
    def _check_model_availability(self, model_id: str) -> bool:
        """Check if model is available locally or can be downloaded."""
        # Check if model exists in cache
        model_path = self.cache_dir / model_id.replace("/", "--")
        if model_path.exists():
            return True
        
        # For now, assume all models can be downloaded
        # In production, you might want to check HuggingFace API
        return True
    
    async def get_all_available_models(self) -> List[Dict[str, Any]]:
        """Get all available models with their status."""
        models = []
        for name, info in self.available_models.items():
            model_data = {
                "name": info.name,
                "model_id": info.model_id,
                "size_gb": info.size_gb,
                "languages": info.languages,
                "device": info.device,
                "model_type": info.model_type,
                "loaded": info.loaded,
                "available": info.available,
                "recommended": info.recommended,
                "mobile_optimized": info.mobile_optimized,
                "accuracy": info.accuracy,
                "description": info.description
            }
            models.append(model_data)
        
        return models
    
    async def get_model_availability(self, model_name: str) -> Dict[str, Any]:
        """Get availability status for a specific model."""
        if model_name not in self.available_models:
            return {"error": f"Model {model_name} not found"}
        
        info = self.available_models[model_name]
        return {
            "name": info.name,
            "available": info.available,
            "loaded": info.loaded,
            "device": info.device,
            "size_gb": info.size_gb,
            "languages": info.languages
        }
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model with hot-switching."""
        if model_name not in self.available_models:
            return {"error": f"Model {model_name} not found"}
        
        model_info = self.available_models[model_name]
        model_type = model_info.model_type
        
        # Use appropriate lock for concurrent safety
        async with self._load_locks[model_type]:
            try:
                # Unload current model of the same type if different
                current_model = None
                current_model_id = None
                if model_type == "chat":
                    current_model = self.current_chat_model
                    current_model_id = current_model.model_id if current_model else None
                elif model_type == "ai_detector":
                    current_model = self.current_ai_detector
                    current_model_id = current_model.metadata.model_id if current_model else None
                
                if current_model and current_model_id and current_model_id != model_info.model_id:
                    await current_model.unload()
                    # Update loaded status for previous model
                    for info in self.available_models.values():
                        if info.model_id == current_model_id and info.model_type == model_type:
                            info.loaded = False
                            break
                
                # Load new model
                model_instance = self.model_instances[model_name]
                success = await model_instance.load()
                
                if success:
                    # Update current model reference
                    if model_type == "chat":
                        self.current_chat_model = model_instance
                    elif model_type == "ai_detector":
                        self.current_ai_detector = model_instance
                    
                    model_info.loaded = True
                    
                    logger.info(f"Successfully loaded {model_type} model: {model_name}")
                    return {
                        "success": True,
                        "model_name": model_name,
                        "model_id": model_info.model_id,
                        "model_type": model_type,
                        "device": model_info.device
                    }
                else:
                    return {"error": f"Failed to load model {model_name}"}
                    
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                return {"error": f"Error loading model: {str(e)}"}
    
    async def unload_current_model(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Unload the currently loaded model. Intelligently determines model type if not specified."""
        
        # Smart backwards compatibility: if no model_type specified, unload any loaded model
        if model_type is None:
            # Legacy behavior: unload whatever is currently loaded
            if self.current_chat_model:
                model_type = "chat"
                current_model = self.current_chat_model
            elif self.current_ai_detector:
                model_type = "ai_detector" 
                current_model = self.current_ai_detector
            else:
                return {"message": "No model currently loaded"}
        else:
            # Explicit model type specified
            if model_type == "chat":
                current_model = self.current_chat_model
            elif model_type == "ai_detector":
                current_model = self.current_ai_detector
            else:
                return {"error": f"Unknown model type: {model_type}"}
            
            if not current_model:
                return {"message": f"No {model_type} model currently loaded"}
        
        try:
            await current_model.unload()
            
            # Update loaded status - handle both chat models and detectors
            if model_type == "chat":
                current_model_id = getattr(current_model, 'model_id', None)
            elif model_type == "ai_detector":
                current_model_id = getattr(current_model, 'metadata', None)
                current_model_id = getattr(current_model_id, 'model_id', None) if current_model_id else None
            else:
                current_model_id = None
                
            if current_model_id:
                for info in self.available_models.values():
                    if info.model_id == current_model_id and info.model_type == model_type:
                        info.loaded = False
                        break
            
            # Clear reference
            if model_type == "chat":
                self.current_chat_model = None
            elif model_type == "ai_detector":
                self.current_ai_detector = None
            
            logger.info(f"Successfully unloaded current {model_type} model")
            return {"success": True, "message": "Model unloaded"}
            
        except Exception as e:
            logger.error(f"Error unloading {model_type} model: {e}")
            return {"error": f"Error unloading model: {str(e)}"}
    
    async def generate_response(
        self,
        prompt: str,
        max_length: int = 4096,  # Increased default to match model improvements
        temperature: float = 0.8,  # Updated to match model defaults
        top_p: float = 0.95,  # Updated to match model defaults
        language: str = "auto",
        system_prompt: Optional[str] = None,  # Added for backwards compatibility
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """Generate response using the currently loaded chat model."""
        logger.info(f"ModelManager.generate_response called with prompt='{prompt}', max_length={max_length}")
        
        if not self.current_chat_model:
            logger.error("No chat model loaded in ModelManager")
            yield "No model loaded. Please load a model first."  # Simple string for backwards compatibility
            return
        
        logger.info(f"Current chat model: {self.current_chat_model.model_id}")
        
        # Use semaphore for concurrency control
        async with self._generation_semaphore:
            try:
                response_chunks = []
                chunk_count = 0
                logger.info("Starting model generation...")
                
                async for chunk in self.current_chat_model.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    language=language,
                    system_prompt=system_prompt,  # Pass through if provided
                    conversation_history=conversation_history  # Pass conversation history
                ):
                    chunk_count += 1
                    response_chunks.append(chunk)
                    yield chunk
                
                logger.info(f"ModelManager generation complete. Total chunks: {chunk_count}")
                
            except Exception as e:
                logger.error(f"Error in ModelManager.generate_response: {e}")
                import traceback
                logger.error(f"ModelManager traceback: {traceback.format_exc()}")
                yield f"Error generating response: {str(e)}"
    
    async def get_current_model_info(self, model_type: str = "chat") -> Dict[str, Any]:
        """Get information about the currently loaded model of specified type."""
        current_model = None
        if model_type == "chat":
            current_model = self.current_chat_model
        elif model_type == "ai_detector":
            current_model = self.current_ai_detector
        
        if not current_model:
            if model_type == "chat":
                return {"error": "No model currently loaded"}  # Legacy message format
            return {"error": f"No {model_type} model currently loaded"}
        
        return await current_model.get_model_info()
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system and model information."""
        result = {
            "hardware": self.hardware_detector.get_system_info(),
            "available_models": await self.get_all_available_models(),
        }
        
        # Backwards compatibility: include current_model field for legacy tests
        if self.current_chat_model:
            result["current_model"] = await self.get_current_model_info("chat")
        else:
            result["current_model"] = None
        
        # New fields for extended functionality
        result["current_chat_model"] = await self.get_current_model_info("chat") if self.current_chat_model else None
        result["current_ai_detector"] = await self.get_current_model_info("ai_detector") if self.current_ai_detector else None
        
        return result
    
    async def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Hot-switch to a different model."""
        return await self.load_model(model_name)
    
    def get_recommended_model(self, model_type: str = "chat") -> Optional[str]:
        """Get the recommended model for current hardware and type."""
        for name, info in self.available_models.items():
            if info.model_type == model_type and info.recommended and info.available:
                return name
        return None
    
    async def auto_load_best_model(self, model_type: str = "chat") -> Dict[str, Any]:
        """Automatically load the best model for current hardware and type."""
        recommended = self.get_recommended_model(model_type)
        if recommended:
            return await self.load_model(recommended)
        else:
            return {"error": f"No suitable {model_type} model found for current hardware"}
    
    # AI Detection Methods
    async def detect_ai_text(
        self, 
        text: str, 
        detector_name: Optional[str] = None,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Detect if text is AI-generated using loaded AI detector models.
        
        Args:
            text: Text to analyze
            detector_name: Specific detector to use (optional, defaults to current)
            return_probabilities: Whether to return detailed probabilities
        """
        try:
            import time
            start_time = time.time()
            
            # Load specific detector if requested and not current
            if detector_name and detector_name in self.available_models:
                model_info = self.available_models[detector_name]
                if model_info.model_type == "ai_detector":
                    if not model_info.loaded:
                        load_result = await self.load_model(detector_name)
                        if not load_result.get("success"):
                            return {
                                "success": False,
                                "error": f"Failed to load detector: {load_result.get('error')}",
                                "is_ai_generated": False,
                                "confidence": 0.0
                            }
            
            # Use current AI detector
            if not self.current_ai_detector:
                # Try to auto-load best AI detector
                auto_load_result = await self.auto_load_best_model("ai_detector")
                if not auto_load_result.get("success"):
                    return {
                        "success": False,
                        "error": "No AI detector loaded and failed to auto-load",
                        "is_ai_generated": False,
                        "confidence": 0.0
                    }
            
            # Use the new detector interface with semaphore for concurrency control
            async with self._detection_semaphore:
                detection_result = await self.current_ai_detector.detect_async(text)
                
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "is_ai_generated": detection_result.is_ai_generated,
                    "confidence": detection_result.confidence,
                    "ai_probability": detection_result.ai_probability,
                    "human_probability": 1.0 - detection_result.ai_probability,
                    "model": detection_result.model_name,
                    "text_length": len(text),
                    "chunks_processed": 1,  # New interface processes as single chunk
                    "detection_method": detection_result.method,
                    "processing_time": processing_time,
                    "chunk_results": [detection_result] if return_probabilities else None
                }
                
        except Exception as e:
            logger.error(f"Error in AI text detection: {e}")
            import traceback
            logger.error(f"AI detection traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "is_ai_generated": False,
                "confidence": 0.0
            }
    
    async def get_ai_detectors(self) -> Dict[str, Any]:
        """Get list of available AI detection models."""
        try:
            ai_models = [
                {
                    "name": info.name,
                    "model_id": info.model_id,
                    "loaded": info.loaded,
                    "available": info.available,
                    "recommended": info.recommended,
                    "device": info.device,
                    "size_gb": info.size_gb,
                    "accuracy": info.accuracy,
                    "description": info.description,
                    "languages": info.languages
                }
                for info in self.available_models.values() 
                if info.model_type == "ai_detector"
            ]
            
            return {
                "success": True,
                "detectors": ai_models,
                "current_detector": await self.get_current_model_info("ai_detector") if self.current_ai_detector else None
            }
        except Exception as e:
            logger.error(f"Error getting AI detectors: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_ai_detector(self, detector_name: str, detector_type: str = "local", **kwargs) -> Dict[str, Any]:
        """Add a new AI detector to the registry."""
        try:
            # This would be implemented to add new detectors dynamically
            # For now, return success as the registry handles known detectors
            return {
                "success": True,
                "message": f"Detector registry manages known detectors automatically",
                "detector_name": detector_name
            }
        except Exception as e:
            logger.error(f"Error adding AI detector: {e}")
            return {
                "success": False,
                "error": str(e)
            }

