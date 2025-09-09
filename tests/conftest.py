"""
Pytest configuration and fixtures for Local LLM Service tests.
"""

import pytest
import asyncio

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from models.manager import ModelManager
from models.detector import HardwareDetector
from models.qwen import QwenModel


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_hardware_specs():
    """Mock hardware specifications for testing."""
    return {
        "cpu_cores": 8,
        "total_memory_gb": 16.0,
        "available_memory_gb": 12.0,
        "has_gpu": True,
        "gpu_memory_gb": 8.0,
        "gpu_name": "NVIDIA RTX 3060",
        "platform": "windows",
        "architecture": "x86_64"
    }


@pytest.fixture
def mock_hardware_detector(mock_hardware_specs):
    """Mock hardware detector for testing."""
    with patch('models.detector.HardwareDetector._detect_hardware') as mock_detect:
        mock_detect.return_value = Mock(**mock_hardware_specs)
        detector = HardwareDetector()
        detector.specs = Mock(**mock_hardware_specs)
        yield detector


@pytest.fixture
def mock_model_manager(temp_cache_dir, mock_hardware_detector):
    """Mock model manager for testing."""
    with patch('models.manager.HardwareDetector') as mock_detector_class:
        mock_detector_class.return_value = mock_hardware_detector
        manager = ModelManager(cache_dir=str(temp_cache_dir))
        yield manager


@pytest.fixture
def mock_qwen_model(temp_cache_dir):
    """Mock Qwen model for testing."""
    model = QwenModel(
        model_id="Qwen/Qwen-1.8B-Chat",
        cache_dir=str(temp_cache_dir),
        device="cpu"
    )
    # Mock the model and tokenizer
    model.model = Mock()
    model.tokenizer = Mock()
    model._loaded = True
    yield model


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's 2+2?"}
    ]


@pytest.fixture
def sample_model_info():
    """Sample model information for testing."""
    return {
        "name": "Qwen-1.8B-Chat",
        "model_id": "Qwen/Qwen-1.8B-Chat",
        "size_gb": 3.6,
        "languages": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
        "device": "cpu",
        "loaded": False,
        "available": True,
        "recommended": True,
        "mobile_optimized": False
    }


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API testing."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.close = AsyncMock()
    yield websocket


@pytest.fixture
def test_app():
    """Test FastAPI app instance."""
    from service.main import app
    return app


@pytest.fixture
def test_client(test_app):
    """Test client for FastAPI app."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def mock_torch():
    """Mock torch for testing."""
    with patch('torch.cuda.is_available') as mock_cuda:
        with patch('torch.cuda.get_device_properties') as mock_props:
            with patch('torch.cuda.get_device_name') as mock_name:
                mock_cuda.return_value = True
                mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
                mock_name.return_value = "NVIDIA RTX 3060"
                yield {
                    'cuda_available': mock_cuda,
                    'device_properties': mock_props,
                    'device_name': mock_name
                }


@pytest.fixture
def mock_transformers():
    """Mock transformers library for testing."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer:
        with patch('transformers.AutoModelForCausalLM') as mock_model:
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            yield {
                'tokenizer': mock_tokenizer,
                'model': mock_model
            }


@pytest.fixture
def mock_psutil():
    """Mock psutil for hardware detection testing."""
    with patch('psutil.cpu_count') as mock_cpu:
        with patch('psutil.virtual_memory') as mock_memory:
            mock_cpu.return_value = 8
            mock_memory.return_value.total = 16 * 1024**3  # 16GB
            mock_memory.return_value.available = 12 * 1024**3  # 12GB
            yield {
                'cpu_count': mock_cpu,
                'virtual_memory': mock_memory
            }


@pytest.fixture
def mock_platform():
    """Mock platform module for testing."""
    with patch('platform.system') as mock_system:
        with patch('platform.machine') as mock_machine:
            mock_system.return_value = "Windows"
            mock_machine.return_value = "AMD64"
            yield {
                'system': mock_system,
                'machine': mock_machine
            }
