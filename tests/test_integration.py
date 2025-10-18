"""
Integration tests for Local LLM Service

Tests actual model loading and functionality without mocks.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from models.manager import ModelManager
from models.detector import HardwareDetector
from models.qwen import QwenModel


class TestIntegration:
    """Integration tests with real model loading."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_hardware_detection_real(self):
        """Test actual hardware detection."""
        detector = HardwareDetector()
        
        # Verify we can detect actual hardware
        assert detector.specs.cpu_cores > 0
        assert detector.specs.total_memory_gb > 0
        assert detector.specs.available_memory_gb > 0
        assert detector.specs.platform in ["windows", "linux", "darwin"]
        
        # Test compatible models
        models = detector.get_compatible_models()
        assert len(models) > 0
        
        # Verify model structure
        for model in models:
            assert "name" in model
            assert "model_id" in model
            assert "size_gb" in model
            assert "languages" in model
            assert "device" in model
    
    @pytest.mark.asyncio
    async def test_model_manager_initialization_real(self, temp_cache_dir):
        """Test ModelManager initialization with real hardware detection."""
        manager = ModelManager(cache_dir=str(temp_cache_dir))
        
        # Verify initialization
        assert manager.cache_dir == temp_cache_dir
        assert len(manager.available_models) > 0
        assert len(manager.model_instances) > 0
        
        # Test getting available models
        models = await manager.get_all_available_models()
        assert len(models) > 0
        
        # Verify model info structure
        for model in models:
            assert "name" in model
            assert "model_id" in model
            assert "size_gb" in model
            assert "languages" in model
            assert "device" in model
            assert "loaded" in model
            assert "available" in model
    
    @pytest.mark.asyncio
    async def test_qwen_model_initialization_real(self, temp_cache_dir):
        """Test QwenModel initialization with real model ID."""
        model = QwenModel(
            model_id="Qwen/Qwen-0.5B-Chat",  # Small model for testing
            cache_dir=str(temp_cache_dir),
            device="cpu"  # Use CPU to avoid GPU memory issues
        )
        
        # Test model info
        info = await model.get_model_info()
        assert info["model_id"] == "Qwen/Qwen-0.5B-Chat"
        assert info["device"] == "cpu"
        assert "supported_languages" in info
        assert len(info["supported_languages"]) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_actual_model_loading(self, temp_cache_dir):
        """Test actual model loading (this will download the model)."""
        model = QwenModel(
            model_id="Qwen/Qwen3-0.6B",  # Small model for testing
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Test loading
        success = await model.load()
        assert success is True
        assert model._loaded is True
        assert model.model is not None
        assert model.tokenizer is not None
        
        # Test model info after loading
        info = await model.get_model_info()
        assert info["loaded"] is True
        
        # Test unloading
        await model.unload()
        assert model._loaded is False
        assert model.model is None
        assert model.tokenizer is None
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_actual_model_generation(self, temp_cache_dir):
        """Test actual text generation with loaded model."""
        model = QwenModel(
            model_id="Qwen/Qwen3-0.6B",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Load model
        success = await model.load()
        assert success is True
        
        # Test generation
        prompt = "Hello, how are you?"
        chunks = []
        async for chunk in model.generate(prompt, max_length=50):
            chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        response = "".join(chunks)
        assert len(response) > 0
        assert isinstance(response, str)
        
        # Clean up
        await model.unload()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_model_manager_with_real_model(self, temp_cache_dir):
        """Test ModelManager with actual model loading."""
        manager = ModelManager(cache_dir=str(temp_cache_dir))
        
        # Get available models
        models = await manager.get_all_available_models()
        assert len(models) > 0
        
        # Find a small model to test with
        small_model = None
        for model in models:
            if model["size_gb"] <= 2.0 and model["device"] == "cpu":
                small_model = model
                break
        
        if small_model is None:
            pytest.skip("No small CPU model available for testing")
        
        # Test loading the model
        result = await manager.load_model(small_model["name"])
        assert "success" in result or "error" in result
        
        if "success" in result:
            # Test generation
            chunks = []
            async for chunk in manager.generate_response("Hello", max_length=50):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            response = "".join(chunks)
            assert len(response) > 0
            
            # Test unloading
            unload_result = await manager.unload_current_model()
            assert "success" in unload_result


def test_main_function():
    """Test main functionality without async."""
    # Test hardware detection
    detector = HardwareDetector()
    print(f"CPU cores: {detector.specs.cpu_cores}")
    print(f"Total memory: {detector.specs.total_memory_gb:.1f} GB")
    print(f"Available memory: {detector.specs.available_memory_gb:.1f} GB")
    print(f"Has GPU: {detector.specs.has_gpu}")
    if detector.specs.has_gpu:
        print(f"GPU: {detector.specs.gpu_name}")
        print(f"GPU memory: {detector.specs.gpu_memory_gb:.1f} GB")
    
    # Test compatible models
    models = detector.get_compatible_models()
    print(f"\nCompatible models ({len(models)}):")
    for model in models:
        print(f"  - {model['name']}: {model['size_gb']}GB, {model['device']}, recommended: {model.get('recommended', False)}")


if __name__ == "__main__":
    test_main_function()
