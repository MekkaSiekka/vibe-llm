"""
Unit tests for QwenModel class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from models.qwen import QwenModel


class TestQwenModel:
    """Test QwenModel class."""
    
    def test_qwen_model_initialization(self, temp_cache_dir):
        """Test QwenModel initialization."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        assert model.model_id == "Qwen/Qwen-1.8B-Chat"
        assert model.cache_dir == str(temp_cache_dir)
        assert model.device == "cpu"
        assert model.model is None
        assert model.tokenizer is None
        assert model._loaded is False
        assert model.executor is not None
    
    def test_qwen_model_language_codes(self, temp_cache_dir):
        """Test language codes mapping."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        assert "en" in model.language_codes
        assert "zh" in model.language_codes
        assert "fr" in model.language_codes
        assert "de" in model.language_codes
        assert "es" in model.language_codes
        assert "ru" in model.language_codes
        assert "ja" in model.language_codes
        assert "ko" in model.language_codes
        
        assert model.language_codes["en"] == "English"
        assert model.language_codes["zh"] == "中文"
    
    @patch('models.qwen.AutoTokenizer')
    @patch('models.qwen.AutoModelForCausalLM')
    @patch('models.qwen.torch.cuda.is_available')
    def test_load_model_cpu(self, mock_cuda, mock_model_class, mock_tokenizer_class, temp_cache_dir):
        """Test model loading on CPU."""
        mock_cuda.return_value = False
        
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Test loading
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(model.load())
            assert result is True
            assert model._loaded is True
            assert model.tokenizer == mock_tokenizer
            assert model.model == mock_model
        finally:
            loop.close()
    
    @patch('models.qwen.AutoTokenizer')
    @patch('models.qwen.AutoModelForCausalLM')
    @patch('models.qwen.torch.cuda.is_available')
    @patch('models.qwen.BitsAndBytesConfig')
    def test_load_model_gpu(self, mock_quant_config, mock_cuda, mock_model_class, mock_tokenizer_class, temp_cache_dir):
        """Test model loading on GPU with quantization."""
        mock_cuda.return_value = True
        
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cuda"
        )
        
        # Test loading
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(model.load())
            assert result is True
            assert model._loaded is True
            assert model.tokenizer == mock_tokenizer
            assert model.model == mock_model
        finally:
            loop.close()
    
    @patch('models.qwen.AutoTokenizer')
    @patch('models.qwen.AutoModelForCausalLM')
    def test_load_model_failure(self, mock_model_class, mock_tokenizer_class, temp_cache_dir):
        """Test model loading failure."""
        # Mock tokenizer to raise exception
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Download failed")
        
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Test loading failure
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(model.load())
            assert result is False
            assert model._loaded is False
        finally:
            loop.close()
    
    def test_format_prompt_auto(self, temp_cache_dir):
        """Test prompt formatting with auto language."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        prompt = "Hello, how are you?"
        formatted = model._format_prompt(prompt, "auto")
        
        assert formatted == prompt
    
    def test_format_prompt_specific_language(self, temp_cache_dir):
        """Test prompt formatting with specific language."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        prompt = "Hello, how are you?"
        formatted = model._format_prompt(prompt, "zh")
        
        assert "中文" in formatted
        assert prompt in formatted
    
    def test_format_prompt_unknown_language(self, temp_cache_dir):
        """Test prompt formatting with unknown language."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        prompt = "Hello, how are you?"
        formatted = model._format_prompt(prompt, "unknown")
        
        assert formatted == prompt
    
    @patch('models.qwen.AutoTokenizer')
    @patch('models.qwen.AutoModelForCausalLM')
    def test_generate_not_loaded(self, mock_model_class, mock_tokenizer_class, temp_cache_dir):
        """Test generation when model is not loaded."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Mock the load method to fail
        model.load = AsyncMock(return_value=False)
        
        # Test generation without loading
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def test_generate():
                chunks = []
                async for chunk in model.generate("Hello"):
                    chunks.append(chunk)
                return chunks
            
            chunks = loop.run_until_complete(test_generate())
            assert len(chunks) == 1
            assert "Error: Model not loaded" in chunks[0]
        finally:
            loop.close()
    
    @patch('models.qwen.AutoTokenizer')
    @patch('models.qwen.AutoModelForCausalLM')
    def test_generate_loaded(self, mock_model_class, mock_tokenizer_class, temp_cache_dir):
        """Test generation when model is loaded."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {"input_ids": Mock(shape=[1, 10])}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.shape = [1, 20]
        mock_outputs.__getitem__ = Mock(return_value=Mock())
        mock_outputs.__getitem__.return_value.__getitem__ = Mock(return_value=Mock())
        mock_model.generate.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Mock the model as loaded
        model._loaded = True
        model.tokenizer = mock_tokenizer
        model.model = mock_model
        
        # Mock tokenizer decode
        mock_tokenizer.decode.return_value = "Hello world test response"
        
        # Test generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def test_generate():
                chunks = []
                async for chunk in model.generate("Hello"):
                    chunks.append(chunk)
                return chunks
            
            chunks = loop.run_until_complete(test_generate())
            assert len(chunks) > 0
            assert any("Hello" in chunk or "world" in chunk or "test" in chunk or "response" in chunk for chunk in chunks)
        finally:
            loop.close()
    
    @patch('models.qwen.AutoTokenizer')
    @patch('models.qwen.AutoModelForCausalLM')
    def test_generate_error(self, mock_model_class, mock_tokenizer_class, temp_cache_dir):
        """Test generation error handling."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {"input_ids": Mock(shape=[1, 10])}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.side_effect = Exception("Generation failed")
        mock_model_class.from_pretrained.return_value = mock_model
        
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Mock the model as loaded
        model._loaded = True
        model.tokenizer = mock_tokenizer
        model.model = mock_model
        
        # Test generation error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def test_generate():
                chunks = []
                async for chunk in model.generate("Hello"):
                    chunks.append(chunk)
                return chunks
            
            chunks = loop.run_until_complete(test_generate())
            assert len(chunks) == 1
            assert "Error: Generation failed" in chunks[0]
        finally:
            loop.close()
    
    def test_get_model_info(self, temp_cache_dir):
        """Test model information retrieval."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            info = loop.run_until_complete(model.get_model_info())
            
            assert info["model_id"] == "Qwen/Qwen-1.8B-Chat"
            assert info["loaded"] is False
            assert info["device"] == "cpu"
            assert "en" in info["supported_languages"]
            assert "zh" in info["supported_languages"]
            assert info["cache_dir"] == str(temp_cache_dir)
        finally:
            loop.close()
    
    @patch('models.qwen.torch.cuda.is_available')
    @patch('models.qwen.torch.cuda.empty_cache')
    def test_unload_model(self, mock_empty_cache, mock_cuda, temp_cache_dir):
        """Test model unloading."""
        mock_cuda.return_value = True
        
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Mock loaded model
        model._loaded = True
        model.model = Mock()
        model.tokenizer = Mock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(model.unload())
            
            assert model._loaded is False
            assert model.model is None
            assert model.tokenizer is None
            mock_empty_cache.assert_called_once()
        finally:
            loop.close()
    
    def test_unload_model_no_cuda(self, temp_cache_dir):
        """Test model unloading without CUDA."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Mock loaded model
        model._loaded = True
        model.model = Mock()
        model.tokenizer = Mock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(model.unload())
            
            assert model._loaded is False
            assert model.model is None
            assert model.tokenizer is None
        finally:
            loop.close()
    
    def test_model_cleanup(self, temp_cache_dir):
        """Test model cleanup on deletion."""
        model = QwenModel(
            model_id="Qwen/Qwen-1.8B-Chat",
            cache_dir=str(temp_cache_dir),
            device="cpu"
        )
        
        # Mock executor
        mock_executor = Mock()
        model.executor = mock_executor
        
        # Delete model
        del model
        
        # Executor should be shut down
        mock_executor.shutdown.assert_called_once_with(wait=False)
