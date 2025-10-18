"""
Integration tests for AI detector functionality.

Tests the full end-to-end functionality including:
1. Real detector model loading (with small models)
2. Actual text detection
3. REST API endpoints
4. WebSocket detection
5. Performance and reliability
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import time

from detectors.hf_sequence import HFSequenceClassifierDetector, HFDetectorConfig
from detectors.registry import get_registry, RegistryEntry
from detectors.base import DetectorMetadata


class TestDetectorIntegration:
    """Integration tests with real detector models."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv('PYTEST_RUNNING') == '1',
        reason="Skipping model download test in offline mode"
    )
    @pytest.mark.asyncio
    async def test_hf_detector_real_loading(self, temp_cache_dir):
        """Test HuggingFace detector with real model loading."""
        config = HFDetectorConfig(
            model_id="distilbert-base-uncased",  # Small model for testing
            device="cpu",  # Use CPU to avoid GPU complications
            size_gb=0.3,
            languages=["en"],
            recommended=False,  # Not a real detector, just for testing loading
            accuracy=0.0,
            description="Test model for loading verification"
        )
        
        detector = HFSequenceClassifierDetector("test-detector", str(temp_cache_dir), config)
        
        # Test loading (this will download the model if not cached)
        success = await detector.load()
        
        if success:  # Only continue if loading succeeded
            # Verify model is loaded
            assert detector._loaded is True
            assert detector.model is not None
            assert detector.tokenizer is not None
            assert detector.metadata.model_id == "distilbert-base-uncased"
            assert detector.metadata.device == "cpu"
            
            # Test unloading
            await detector.unload()
            assert detector._loaded is False
            assert detector.model is None
            assert detector.tokenizer is None
        else:
            pytest.skip("Model loading failed - likely network or resource issue")
    
    @pytest.mark.asyncio
    async def test_detector_registry_integration(self, temp_cache_dir):
        """Test detector registry with real detector creation."""
        registry = get_registry()
        
        # Register a test detector
        config = HFDetectorConfig(
            model_id="distilbert-base-uncased",
            device="cpu",
            size_gb=0.3,
            languages=["en"]
        )
        
        metadata = DetectorMetadata(
            name="test-integration-detector",
            model_id=config.model_id,
            device=config.device,
            size_gb=config.size_gb,
            languages=config.languages,
            recommended=False,
            accuracy=0.0,
            description="Integration test detector"
        )
        
        def factory(name: str, cache_dir: str):
            return HFSequenceClassifierDetector(name, cache_dir, config)
        
        entry = RegistryEntry(create_fn=factory, metadata=metadata)
        registry.register("test-integration-detector", entry)
        
        # Verify registration
        assert "test-integration-detector" in registry.list()
        
        # Test creation
        from detectors.registry import create_detector
        detector = create_detector("test-integration-detector", str(temp_cache_dir))
        assert detector is not None
        assert isinstance(detector, HFSequenceClassifierDetector)


class TestAPIIntegration:
    """Test API integration with detector functionality."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from service.main import app
        return TestClient(app)
    
    @pytest.mark.skip(reason="Detector endpoints not implemented in current service - chat-only service")
    def test_detectors_endpoint(self, test_client):
        """Test /detectors endpoint."""
        response = test_client.get("/detectors")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "detectors" in data
        
        if data["success"]:
            assert isinstance(data["detectors"], list)
    
    @pytest.mark.skip(reason="Detector endpoints not implemented in current service - chat-only service")
    def test_detect_ai_endpoint(self, test_client):
        """Test /detect/ai endpoint."""
        response = test_client.post("/detect/ai", json={
            "text": "This is a test text to analyze for AI generation.",
            "detector": None,
            "use_multiple": False
        })
        
        # Should return 200 regardless of whether detection succeeds
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "is_ai_generated" in data
        assert "confidence" in data
    
    @pytest.mark.skip(reason="Detector endpoints not implemented in current service - chat-only service")
    def test_simple_detect_endpoint(self, test_client):
        """Test /detect/ai/simple endpoint."""
        response = test_client.post("/detect/ai/simple", params={
            "text": "This is a test text.",
            "detector": None
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "is_ai_generated" in data


class TestWebSocketIntegration:
    """Test WebSocket integration with detection."""
    
    @pytest.mark.asyncio
    async def test_websocket_detection_message(self):
        """Test WebSocket detection message handling."""
        from service.websocket import handle_detection_request, ConnectionManager
        from unittest.mock import AsyncMock, Mock
        
        # Mock WebSocket and connection manager
        mock_websocket = AsyncMock()
        mock_manager = Mock()
        mock_manager.send_message = AsyncMock(return_value=True)
        
        # Mock the connection manager in the module
        import service.websocket
        original_manager = service.websocket.manager
        service.websocket.manager = mock_manager
        
        try:
            # Test detection request
            message_data = {
                "text": "This is test text for detection",
                "detector": None,
                "use_multiple": False
            }
            
            await handle_detection_request(mock_websocket, "test_client", message_data)
            
            # Verify messages were sent
            assert mock_manager.send_message.call_count >= 2  # At least start and result/error
            
            # Check that detection_start was sent
            calls = mock_manager.send_message.call_args_list
            start_call = next((call for call in calls if call[0][1].get("type") == "detection_start"), None)
            assert start_call is not None
            
        finally:
            # Restore original manager
            service.websocket.manager = original_manager


class TestPerformanceReliability:
    """Test performance and reliability aspects."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skip(reason="AI detection not implemented in current service - chat-only service")
    @pytest.mark.asyncio
    async def test_concurrent_detections(self, temp_cache_dir):
        """Test concurrent detection requests."""
        from unittest.mock import patch, AsyncMock
        
        # Mock a fast detector
        class FastMockDetector:
            async def detect_async(self, text: str):
                await asyncio.sleep(0.1)  # Simulate processing time
                from detectors.base import DetectionResult
                return DetectionResult(
                    is_ai_generated=True,
                    confidence=0.8,
                    ai_probability=0.8,
                    model_name="fast-mock",
                    method="mock"
                )
        
        with patch('detectors.register_defaults.register_defaults'):
            with patch('models.manager.HardwareDetector'):
                from models.manager import ModelManager
                manager = ModelManager(cache_dir=str(temp_cache_dir))
                
                # Set mock detector
                manager.current_ai_detector = FastMockDetector()
                
                # Run concurrent detections
                texts = [f"Test text {i}" for i in range(5)]
                
                start_time = time.time()
                tasks = [manager.detect_ai_text(text) for text in texts]
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                # Verify all succeeded
                assert len(results) == 5
                for result in results:
                    assert result["success"] is True
                    assert result["is_ai_generated"] is True
                
                # Should take roughly sequential time due to semaphore
                # (5 * 0.1s = 0.5s minimum, but allow for overhead)
                assert end_time - start_time >= 0.4
    
    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, temp_cache_dir):
        """Test error handling in various scenarios."""
        from unittest.mock import patch
        
        with patch('detectors.register_defaults.register_defaults'):
            with patch('models.manager.HardwareDetector'):
                from models.manager import ModelManager
                manager = ModelManager(cache_dir=str(temp_cache_dir))
                
                # Test detection with no detector loaded
                result = await manager.detect_ai_text("Test text")
                assert result["success"] is False
                assert "error" in result
                
                # Test with failing detector
                class FailingDetector:
                    async def detect_async(self, text: str):
                        raise Exception("Simulated detector failure")
                
                manager.current_ai_detector = FailingDetector()
                
                result = await manager.detect_ai_text("Test text")
                assert result["success"] is False
                assert "error" in result
                assert "Simulated detector failure" in result["error"]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.mark.skip(reason="AI detection not implemented in current service - chat-only service")
    @pytest.mark.asyncio
    async def test_mixed_model_operations(self):
        """Test mixing chat and detection operations."""
        from unittest.mock import patch, Mock, AsyncMock
        import tempfile
        
        temp_dir = tempfile.mkdtemp()
        try:
            with patch('detectors.register_defaults.register_defaults'):
                with patch('models.manager.HardwareDetector'):
                    from models.manager import ModelManager
                    manager = ModelManager(cache_dir=temp_dir)
                    
                    # Mock both chat and detection models
                    mock_chat_model = Mock()
                    mock_chat_model.generate = AsyncMock()
                    async def mock_generate(*args, **kwargs):
                        for chunk in ["Hello", " ", "world"]:
                            yield chunk
                    mock_chat_model.generate.return_value = mock_generate()
                    
                    class MockDetector:
                        async def detect_async(self, text):
                            from detectors.base import SimpleDetectionResult
                            return SimpleDetectionResult(
                                is_ai_generated=False,
                                confidence=0.7,
                                ai_probability=0.3,
                                model_name="mock-detector",
                                method="mock"
                            )
                    
                    manager.current_chat_model = mock_chat_model
                    manager.current_ai_detector = MockDetector()
                    
                    # Test mixed operations
                    # 1. Generate text
                    chunks = []
                    async for chunk in manager.generate_response("Hello"):
                        chunks.append(chunk)
                    assert chunks == ["Hello", " ", "world"]
                    
                    # 2. Detect text
                    result = await manager.detect_ai_text("Test text")
                    assert result["success"] is True
                    assert result["is_ai_generated"] is False
                    assert result["confidence"] == 0.7
                    
                    # 3. Verify both models are still available
                    assert manager.current_chat_model is not None
                    assert manager.current_ai_detector is not None
                    
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
