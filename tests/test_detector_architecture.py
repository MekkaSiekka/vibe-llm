"""
Comprehensive tests for the new AI detector architecture.

This test suite validates:
1. Base interfaces and dataclasses
2. Registry functionality  
3. HuggingFace detector implementation
4. Integration with ModelManager
5. Concurrency controls
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil
from pathlib import Path

from detectors.base import AIDetector, DetectorMetadata, SimpleDetectionResult
from detectors.registry import DetectorRegistry, RegistryEntry, get_registry
from detectors.hf_sequence import HFSequenceClassifierDetector, HFDetectorConfig


class TestDetectorBase:
    """Test base interfaces and dataclasses."""
    
    def test_detection_result_creation(self):
        """Test SimpleDetectionResult dataclass creation."""
        result = SimpleDetectionResult(
            is_ai_generated=True,
            confidence=0.95,
            ai_probability=0.95,
            model_name="test-model",
            method="transformer_classification"
        )
        
        assert result.is_ai_generated is True
        assert result.confidence == 0.95
        assert result.ai_probability == 0.95
        assert result.model_name == "test-model"
        assert result.method == "transformer_classification"
    
    def test_detector_metadata_creation(self):
        """Test DetectorMetadata dataclass creation."""
        metadata = DetectorMetadata(
            name="test-detector",
            model_id="test/model",
            device="cpu",
            size_gb=0.5,
            languages=["en"],
            recommended=True,
            accuracy=0.95,
            description="Test detector"
        )
        
        assert metadata.name == "test-detector"
        assert metadata.model_id == "test/model"
        assert metadata.device == "cpu"
        assert metadata.size_gb == 0.5
        assert metadata.languages == ["en"]
        assert metadata.recommended is True
        assert metadata.accuracy == 0.95
        assert metadata.description == "Test detector"


class TestDetectorRegistry:
    """Test detector registry functionality."""
    
    def test_registry_creation(self):
        """Test registry can be created."""
        registry = DetectorRegistry()
        assert len(registry.list()) == 0
    
    def test_registry_registration(self):
        """Test detector registration."""
        registry = DetectorRegistry()
        
        # Create mock detector and metadata
        metadata = DetectorMetadata(
            name="test-detector",
            model_id="test/model",
            device="cpu",
            size_gb=0.5,
            languages=["en"]
        )
        
        def mock_factory(name: str, cache_dir: str):
            return Mock(spec=AIDetector)
        
        entry = RegistryEntry(create_fn=mock_factory, metadata=metadata)
        registry.register("test-detector", entry)
        
        # Verify registration
        assert "test-detector" in registry.list()
        assert registry.get("test-detector") is not None
        assert registry.get("nonexistent") is None
    
    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2


class MockDetector(AIDetector):
    """Mock detector for testing."""
    
    def __init__(self, name: str, cache_dir: str):
        self.name = name
        self.cache_dir = cache_dir
        self._loaded = False
    
    async def detect_async(self, text: str) -> SimpleDetectionResult:
        """Mock detection that returns fixed result."""
        return SimpleDetectionResult(
            is_ai_generated=True,
            confidence=0.8,
            ai_probability=0.8,
            model_name=self.name,
            method="mock_classification"
        )
    
    async def load(self) -> bool:
        """Mock loading."""
        self._loaded = True
        return True
    
    async def unload(self):
        """Mock unloading."""
        self._loaded = False
    
    async def get_model_info(self) -> dict:
        """Mock model info."""
        return {
            "model_id": f"mock/{self.name}",
            "loaded": self._loaded,
            "device": "cpu",
            "supported_languages": ["en"]
        }


class TestModelManagerIntegration:
    """Test integration with ModelManager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_hardware_detector(self):
        """Mock hardware detector."""
        with patch('models.detector.HardwareDetector._detect_hardware') as mock_detect:
            mock_specs = Mock(
                cpu_cores=8,
                total_memory_gb=16.0,
                available_memory_gb=12.0,
                has_gpu=True,
                gpu_memory_gb=8.0,
                gpu_name="NVIDIA RTX 3060",
                platform="windows",
                architecture="x86_64"
            )
            mock_detect.return_value = mock_specs
            
            from models.detector import HardwareDetector
            detector = HardwareDetector()
            detector.specs = mock_specs
            yield detector
    
    @pytest.mark.asyncio
    async def test_model_manager_with_detector_registry(self, temp_cache_dir, mock_hardware_detector):
        """Test ModelManager works with new detector registry."""
        
        # Mock the registry to return our mock detector
        with patch('detectors.register_defaults.register_defaults') as mock_register:
            with patch('detectors.registry.create_detector') as mock_create:
                mock_create.return_value = MockDetector("test-detector", str(temp_cache_dir))
                
                # Import after patching
                with patch('models.manager.HardwareDetector') as mock_detector_class:
                    mock_detector_class.return_value = mock_hardware_detector
                    
                    from models.manager import ModelManager
                    manager = ModelManager(cache_dir=str(temp_cache_dir))
                    
                    # Verify manager was created
                    assert manager is not None
                    assert manager.detector_registry is not None
    
    @pytest.mark.asyncio
    async def test_ai_detection_with_new_architecture(self, temp_cache_dir, mock_hardware_detector):
        """Test AI detection using new architecture."""
        
        # Create a mock detector that will be returned by the registry
        mock_detector = MockDetector("test-detector", str(temp_cache_dir))
        await mock_detector.load()
        
        with patch('detectors.register_defaults.register_defaults'):
            with patch('detectors.registry.create_detector') as mock_create:
                mock_create.return_value = mock_detector
                
                with patch('models.manager.HardwareDetector') as mock_detector_class:
                    mock_detector_class.return_value = mock_hardware_detector
                    
                    from models.manager import ModelManager
                    manager = ModelManager(cache_dir=str(temp_cache_dir))
                    
                    # Manually set current detector for testing
                    manager.current_ai_detector = mock_detector
                    
                    # Test detection
                    result = await manager.detect_ai_text("This is test text")
                    
                    assert result["success"] is True
                    assert result["is_ai_generated"] is True
                    assert result["confidence"] == 0.8
                    assert result["ai_probability"] == 0.8
                    assert result["model"] == "test-detector"


class TestConcurrencyControls:
    """Test concurrency controls in ModelManager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_hardware_detector(self):
        """Mock hardware detector."""
        with patch('models.detector.HardwareDetector._detect_hardware') as mock_detect:
            mock_specs = Mock(
                cpu_cores=8,
                total_memory_gb=16.0,
                available_memory_gb=12.0,
                has_gpu=False,  # Use CPU to avoid GPU complications
                gpu_memory_gb=0.0,
                platform="windows",
                architecture="x86_64"
            )
            mock_detect.return_value = mock_specs
            
            from models.detector import HardwareDetector
            detector = HardwareDetector()
            detector.specs = mock_specs
            yield detector
    
    @pytest.mark.asyncio
    async def test_concurrent_loading(self, temp_cache_dir, mock_hardware_detector):
        """Test concurrent model loading is properly serialized."""
        
        with patch('detectors.register_defaults.register_defaults'):
            with patch('models.manager.HardwareDetector') as mock_detector_class:
                mock_detector_class.return_value = mock_hardware_detector
                
                from models.manager import ModelManager
                manager = ModelManager(cache_dir=str(temp_cache_dir))
                
                # Verify load locks exist
                assert "chat" in manager._load_locks
                assert "ai_detector" in manager._load_locks
                assert isinstance(manager._load_locks["chat"], asyncio.Lock)
                assert isinstance(manager._load_locks["ai_detector"], asyncio.Lock)
    
    @pytest.mark.asyncio
    async def test_generation_semaphore(self, temp_cache_dir, mock_hardware_detector):
        """Test generation uses semaphore for concurrency control."""
        
        with patch('detectors.register_defaults.register_defaults'):
            with patch('models.manager.HardwareDetector') as mock_detector_class:
                mock_detector_class.return_value = mock_hardware_detector
                
                from models.manager import ModelManager
                manager = ModelManager(cache_dir=str(temp_cache_dir))
                
                # Verify semaphores exist
                assert isinstance(manager._generation_semaphore, asyncio.Semaphore)
                assert isinstance(manager._detection_semaphore, asyncio.Semaphore)
                
                # Verify semaphore limits
                assert manager._generation_semaphore._value == 1
                assert manager._detection_semaphore._value == 1


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing APIs."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture 
    def mock_hardware_detector(self):
        """Mock hardware detector."""
        with patch('models.detector.HardwareDetector._detect_hardware') as mock_detect:
            mock_specs = Mock(
                cpu_cores=8,
                total_memory_gb=16.0,
                available_memory_gb=12.0,
                has_gpu=False,
                gpu_memory_gb=0.0,
                platform="windows",
                architecture="x86_64"
            )
            mock_detect.return_value = mock_specs
            
            from models.detector import HardwareDetector
            detector = HardwareDetector()
            detector.specs = mock_specs
            yield detector
    
    @pytest.mark.asyncio
    async def test_current_model_property(self, temp_cache_dir, mock_hardware_detector):
        """Test current_model property works for backwards compatibility."""
        
        with patch('detectors.register_defaults.register_defaults'):
            with patch('models.manager.HardwareDetector') as mock_detector_class:
                mock_detector_class.return_value = mock_hardware_detector
                
                from models.manager import ModelManager
                manager = ModelManager(cache_dir=str(temp_cache_dir))
                
                # Initially no model loaded
                assert manager.current_model is None
                
                # Mock loading a detector
                mock_detector = MockDetector("test-detector", str(temp_cache_dir))
                manager.current_ai_detector = mock_detector
                
                # Should return the AI detector
                assert manager.current_model is mock_detector
    
    @pytest.mark.asyncio  
    async def test_intelligent_unload(self, temp_cache_dir, mock_hardware_detector):
        """Test intelligent unload works with any loaded model."""
        
        with patch('detectors.register_defaults.register_defaults'):
            with patch('models.manager.HardwareDetector') as mock_detector_class:
                mock_detector_class.return_value = mock_hardware_detector
                
                from models.manager import ModelManager
                manager = ModelManager(cache_dir=str(temp_cache_dir))
                
                # Mock loading a detector
                mock_detector = MockDetector("test-detector", str(temp_cache_dir))
                await mock_detector.load()
                manager.current_ai_detector = mock_detector
                
                # Should be able to unload without specifying type
                result = await manager.unload_current_model()
                assert result["success"] is True
                assert result["message"] == "Model unloaded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
