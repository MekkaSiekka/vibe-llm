"""
Unit tests for ModelManager class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from models.manager import ModelManager, ModelInfo


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test ModelInfo creation with all parameters."""
        info = ModelInfo(
            name="Qwen-7B-Chat",
            model_id="Qwen/Qwen-7B-Chat",
            size_gb=14.0,
            languages=["en", "zh", "fr"],
            device="cuda",
            loaded=True,
            available=True,
            recommended=True,
            mobile_optimized=False
        )

        assert info.name == "Qwen-7B-Chat"
        assert info.model_id == "Qwen/Qwen-7B-Chat"
        assert info.size_gb == 14.0
        assert info.languages == ["en", "zh", "fr"]
        assert info.device == "cuda"
        assert info.loaded is True
        assert info.available is True
        assert info.recommended is True
        assert info.mobile_optimized is False

    def test_model_info_defaults(self):
        """Test ModelInfo with default values."""
        info = ModelInfo(
            name="Qwen-1.8B-Chat",
            model_id="Qwen/Qwen-1.8B-Chat",
            size_gb=3.6,
            languages=["en", "zh"],
            device="cpu"
        )

        assert info.loaded is False
        assert info.available is False
        assert info.recommended is False
        assert info.mobile_optimized is False


class TestModelManager:
    """Test ModelManager class."""

    def test_model_manager_initialization(self, temp_cache_dir, mock_hardware_detector):
        """Test ModelManager initialization."""
        with patch('models.manager.HardwareDetector') as mock_detector_class:
            mock_detector_class.return_value = mock_hardware_detector
            manager = ModelManager(cache_dir=str(temp_cache_dir))

            assert manager.cache_dir == Path(temp_cache_dir)
            assert manager.current_model is None
            assert len(manager.available_models) > 0
            assert len(manager.model_instances) > 0

    def test_check_model_availability_exists(self, temp_cache_dir, mock_hardware_detector):
        """Test model availability check when model exists."""
        with patch('models.manager.HardwareDetector') as mock_detector_class:
            mock_detector_class.return_value = mock_hardware_detector
            manager = ModelManager(cache_dir=str(temp_cache_dir))

            # Create a mock model directory
            model_path = manager.cache_dir / "Qwen--Qwen-1.8B-Chat"
            model_path.mkdir()

            result = manager._check_model_availability("Qwen/Qwen-1.8B-Chat")
            assert result is True

    def test_check_model_availability_not_exists(self, temp_cache_dir, mock_hardware_detector):
        """Test model availability check when model doesn't exist."""
        with patch('models.manager.HardwareDetector') as mock_detector_class:
            mock_detector_class.return_value = mock_hardware_detector
            manager = ModelManager(cache_dir=str(temp_cache_dir))

            result = manager._check_model_availability("Qwen/Qwen-1.8B-Chat")
            assert result is True  # Assumes download availability

    @pytest.mark.asyncio
    async def test_get_all_available_models(self, mock_model_manager):
        """Test getting all available models."""
        models = await mock_model_manager.get_all_available_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Check structure of first model
        model = models[0]
        expected_keys = ["name", "model_id", "size_gb", "languages", "device",
                        "loaded", "available", "recommended", "mobile_optimized"]
        for key in expected_keys:
            assert key in model

    @pytest.mark.asyncio
    async def test_get_model_availability_existing(self, mock_model_manager):
        """Test getting availability for existing model."""
        # Get first available model name
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]

        result = await mock_model_manager.get_model_availability(model_name)

        assert "name" in result
        assert "available" in result
        assert "loaded" in result
        assert "device" in result
        assert result["name"] == model_name

    @pytest.mark.asyncio
    async def test_get_model_availability_nonexistent(self, mock_model_manager):
        """Test getting availability for non-existent model."""
        result = await mock_model_manager.get_model_availability("NonExistentModel")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_load_model_success(self, mock_model_manager):
        """Test successful model loading."""
        # Get first available model name
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]

        # Mock the model instance load method
        mock_model_instance = mock_model_manager.model_instances[model_name]
        mock_model_instance.load = AsyncMock(return_value=True)

        result = await mock_model_manager.load_model(model_name)

        assert "success" in result
        assert result["success"] is True
        assert result["model_name"] == model_name
        assert mock_model_manager.current_model == mock_model_instance
        assert mock_model_manager.available_models[model_name].loaded is True

    @pytest.mark.asyncio
    async def test_load_model_failure(self, mock_model_manager):
        """Test model loading failure."""
        # Get first available model name
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]

        # Mock the model instance load method to fail
        mock_model_instance = mock_model_manager.model_instances[model_name]
        mock_model_instance.load = AsyncMock(return_value=False)

        result = await mock_model_manager.load_model(model_name)

        assert "error" in result
        assert f"Failed to load model {model_name}" in result["error"]

    @pytest.mark.asyncio
    async def test_load_model_nonexistent(self, mock_model_manager):
        """Test loading non-existent model."""
        result = await mock_model_manager.load_model("NonExistentModel")

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_unload_current_model_success(self, mock_model_manager):
        """Test successful model unloading."""
        # Load a model first
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]
        mock_model_instance = mock_model_manager.model_instances[model_name]
        mock_model_instance.load = AsyncMock(return_value=True)
        mock_model_instance.unload = AsyncMock()

        await mock_model_manager.load_model(model_name)

        # Now unload
        result = await mock_model_manager.unload_current_model()

        assert "success" in result
        assert result["success"] is True
        assert mock_model_manager.current_model is None
        assert mock_model_manager.available_models[model_name].loaded is False

    @pytest.mark.asyncio
    async def test_unload_current_model_no_model(self, mock_model_manager):
        """Test unloading when no model is loaded."""
        result = await mock_model_manager.unload_current_model()

        assert "message" in result
        assert "No model currently loaded" in result["message"]

    @pytest.mark.asyncio
    async def test_generate_response_no_model(self, mock_model_manager):
        """Test response generation when no model is loaded."""
        async def collect_chunks():
            chunks = []
            async for chunk in mock_model_manager.generate_response("Hello"):
                chunks.append(chunk)
            return chunks

        chunks = await collect_chunks()

        assert len(chunks) == 1
        assert "No model loaded" in str(chunks[0])

    @pytest.mark.asyncio
    async def test_generate_response_with_model(self, mock_model_manager):
        """Test response generation with loaded model."""
        # Load a model first
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]
        mock_model_instance = mock_model_manager.model_instances[model_name]
        mock_model_instance.load = AsyncMock(return_value=True)

        await mock_model_manager.load_model(model_name)

        # Mock the generate method as an async generator
        async def mock_generate(*args, **kwargs):
            for chunk in ["Hello", " ", "world"]:
                yield chunk
        
        mock_model_instance.generate = mock_generate

        async def collect_chunks():
            chunks = []
            async for chunk in mock_model_manager.generate_response("Hello"):
                chunks.append(chunk)
            return chunks

        chunks = await collect_chunks()

        assert len(chunks) == 3
        assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_get_current_model_info_no_model(self, mock_model_manager):
        """Test getting current model info when no model is loaded."""
        result = await mock_model_manager.get_current_model_info()

        assert "error" in result
        assert "No model currently loaded" in result["error"]

    @pytest.mark.asyncio
    async def test_get_current_model_info_with_model(self, mock_model_manager):
        """Test getting current model info with loaded model."""
        # Load a model first
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]
        mock_model_instance = mock_model_manager.model_instances[model_name]
        mock_model_instance.load = AsyncMock(return_value=True)
        mock_model_instance.get_model_info = AsyncMock(return_value={"model_id": "test", "loaded": True})

        await mock_model_manager.load_model(model_name)

        result = await mock_model_manager.get_current_model_info()

        assert result["model_id"] == "test"
        assert result["loaded"] is True

    @pytest.mark.asyncio
    async def test_get_system_info(self, mock_model_manager):
        """Test getting comprehensive system information."""
        result = await mock_model_manager.get_system_info()

        assert "hardware" in result
        assert "available_models" in result
        assert "current_model" in result

    def test_get_recommended_model(self, mock_model_manager):
        """Test getting recommended model."""
        recommended = mock_model_manager.get_recommended_model()

        # Should return a model name or None
        if recommended:
            assert isinstance(recommended, str)
            assert recommended in mock_model_manager.available_models

    @pytest.mark.asyncio
    async def test_auto_load_best_model_success(self, mock_model_manager):
        """Test auto-loading the best model."""
        # Get the first available model and make it recommended
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]
        
        # Make it recommended
        mock_model_manager.available_models[model_name].recommended = True
        mock_model_manager.available_models[model_name].available = True

        # Mock the model instance
        mock_model_instance = mock_model_manager.model_instances[model_name]
        mock_model_instance.load = AsyncMock(return_value=True)

        result = await mock_model_manager.auto_load_best_model()

        assert "success" in result
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_auto_load_best_model_no_recommended(self, mock_model_manager):
        """Test auto-loading when no recommended model is available."""
        # Clear recommended models
        for info in mock_model_manager.available_models.values():
            info.recommended = False

        result = await mock_model_manager.auto_load_best_model()

        assert "error" in result
        assert "No suitable model found" in result["error"]

    @pytest.mark.asyncio
    async def test_switch_model_success(self, mock_model_manager):
        """Test model switching."""
        models = await mock_model_manager.get_all_available_models()
        model_name = models[0]["name"]

        # Mock the model instance
        mock_model_instance = mock_model_manager.model_instances[model_name]
        mock_model_instance.load = AsyncMock(return_value=True)

        result = await mock_model_manager.switch_model(model_name)

        assert "success" in result
        assert result["success"] is True
        assert result["model_name"] == model_name
