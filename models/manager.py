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


@dataclass
class ModelInfo:
    """Model information structure."""
    name: str
    model_id: str
    size_gb: float
    languages: List[str]
    device: str
    loaded: bool = False
    available: bool = False
    recommended: bool = False
    mobile_optimized: bool = False


class ModelManager:
    """Manages all available models with hot-switching capabilities."""
    
    def __init__(self, cache_dir: str = "./models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.hardware_detector = HardwareDetector()
        self.current_model: Optional[QwenModel] = None
        self.available_models: Dict[str, ModelInfo] = {}
        self.model_instances: Dict[str, QwenModel] = {}
        
        # Initialize available models
        self._initialize_models()
    
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
                recommended=model_data.get("recommended", False),
                mobile_optimized=model_data.get("mobile_optimized", False),
                available=self._check_model_availability(model_data["model_id"])
            )
            
            self.available_models[model_info.name] = model_info
            
            # Create model instance
            self.model_instances[model_info.name] = QwenModel(
                model_id=model_info.model_id,
                cache_dir=str(self.cache_dir),
                device=model_info.device
            )
        
        logger.info(f"Initialized {len(self.available_models)} compatible models")
    
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
                "loaded": info.loaded,
                "available": info.available,
                "recommended": info.recommended,
                "mobile_optimized": info.mobile_optimized
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
        
        try:
            # Unload current model if different
            if self.current_model and self.current_model.model_id != self.available_models[model_name].model_id:
                await self.current_model.unload()
                # Update loaded status for previous model
                for info in self.available_models.values():
                    if info.model_id == self.current_model.model_id:
                        info.loaded = False
                        break
            
            # Load new model
            model_instance = self.model_instances[model_name]
            success = await model_instance.load()
            
            if success:
                self.current_model = model_instance
                self.available_models[model_name].loaded = True
                
                logger.info(f"Successfully loaded model: {model_name}")
                return {
                    "success": True,
                    "model_name": model_name,
                    "model_id": self.available_models[model_name].model_id,
                    "device": self.available_models[model_name].device
                }
            else:
                return {"error": f"Failed to load model {model_name}"}
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {"error": f"Error loading model: {str(e)}"}
    
    async def unload_current_model(self) -> Dict[str, Any]:
        """Unload the currently loaded model."""
        if not self.current_model:
            return {"message": "No model currently loaded"}
        
        try:
            await self.current_model.unload()
            
            # Update loaded status
            for info in self.available_models.values():
                if info.loaded:
                    info.loaded = False
                    break
            
            self.current_model = None
            logger.info("Successfully unloaded current model")
            return {"success": True, "message": "Model unloaded"}
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return {"error": f"Error unloading model: {str(e)}"}
    
    async def generate_response(
        self, 
        prompt: str, 
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        language: str = "auto"
    ):
        """Generate response using the currently loaded model."""
        logger.info(f"ModelManager.generate_response called with prompt='{prompt}', max_length={max_length}")
        
        if not self.current_model:
            logger.error("No model loaded in ModelManager")
            yield {"error": "No model loaded. Please load a model first."}
            return
        
        logger.info(f"Current model: {self.current_model.model_id}")
        
        try:
            response_chunks = []
            chunk_count = 0
            logger.info("Starting model generation...")
            
            async for chunk in self.current_model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                language=language
            ):
                chunk_count += 1
                logger.info(f"ModelManager received chunk #{chunk_count}: {repr(chunk)}")
                response_chunks.append(chunk)
                yield chunk
            
            logger.info(f"ModelManager generation complete. Total chunks: {chunk_count}")
            
        except Exception as e:
            logger.error(f"Error in ModelManager.generate_response: {e}")
            import traceback
            logger.error(f"ModelManager traceback: {traceback.format_exc()}")
            yield f"Error: {str(e)}"
    
    async def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if not self.current_model:
            return {"error": "No model currently loaded"}
        
        return await self.current_model.get_model_info()
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system and model information."""
        return {
            "hardware": self.hardware_detector.get_system_info(),
            "available_models": await self.get_all_available_models(),
            "current_model": await self.get_current_model_info() if self.current_model else None
        }
    
    async def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Hot-switch to a different model."""
        return await self.load_model(model_name)
    
    def get_recommended_model(self) -> Optional[str]:
        """Get the recommended model for current hardware."""
        for name, info in self.available_models.items():
            if info.recommended and info.available:
                return name
        return None
    
    async def auto_load_best_model(self) -> Dict[str, Any]:
        """Automatically load the best model for current hardware."""
        recommended = self.get_recommended_model()
        if recommended:
            return await self.load_model(recommended)
        else:
            return {"error": "No suitable model found for current hardware"}

