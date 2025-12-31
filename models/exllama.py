"""
ExLlamaV2 Model Integration

Handles EXL2 quantized model loading, inference, and management using ExLlamaV2 library.
"""

import os
import torch
from typing import Dict, List, Optional, AsyncGenerator, Any
from pathlib import Path
from loguru import logger
import asyncio

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
    EXLLAMAV2_AVAILABLE = True
except ImportError:
    EXLLAMAV2_AVAILABLE = False
    logger.warning("ExLlamaV2 not available. Install with: pip install exllamav2")


class ExLlamaModel:
    """ExLlamaV2 model wrapper with async support for EXL2 quantized models."""
    
    def __init__(self, model_id: str, cache_dir: str, device: str = "cuda", precision_mode: Optional[str] = None):
        """
        Initialize ExLlamaModel.
        
        Args:
            model_id: HuggingFace model ID (e.g., "Dracones/Llama-3.3-70B-Instruct_exl2_2.5bpw")
            cache_dir: Directory to cache downloaded models
            device: Device to use (only "cuda" supported for ExLlamaV2)
            precision_mode: Ignored for EXL2 models (quantization is pre-baked)
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self.precision_mode = precision_mode  # Not used for EXL2, but kept for interface compatibility
        
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.generator = None
        self._loaded = False
        
        # Language support (primarily English for Llama models)
        self.language_codes = {
            "en": "English",
        }
        
        # Context window settings
        self.max_context_tokens = 8192
        self.preferred_context_tokens = 4096
        self.min_context_tokens = 512
        self.context_window_safety_margin = 256
        self.max_history_messages = 50
        self.conversation_history = []
        self._context_token_count = 0
    
    def _get_model_path(self) -> Path:
        """Get the local path to the model, downloading if necessary."""
        # Convert HuggingFace model ID to local path
        model_dir_name = self.model_id.replace("/", "--")
        model_path = Path(self.cache_dir) / model_dir_name
        
        if not model_path.exists():
            # Download from HuggingFace
            logger.info(f"Downloading ExLlamaV2 model: {self.model_id}")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=self.model_id,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False
                )
                logger.info(f"Downloaded model to: {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model {self.model_id}: {e}")
                raise
        
        return model_path
    
    async def load(self) -> bool:
        """Load the ExLlamaV2 model asynchronously."""
        if self._loaded:
            return True
        
        if not EXLLAMAV2_AVAILABLE:
            logger.error("ExLlamaV2 library not installed. Install with: pip install exllamav2")
            return False
        
        if self.device != "cuda" or not torch.cuda.is_available():
            logger.error("ExLlamaV2 requires CUDA. GPU not available.")
            return False
        
        try:
            logger.info(f"Loading ExLlamaV2 model: {self.model_id}")
            
            # Run model loading in thread pool to avoid blocking
            await asyncio.to_thread(self._load_model_sync)
            
            self._loaded = True
            logger.info(f"Successfully loaded ExLlamaV2 model: {self.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ExLlamaV2 model {self.model_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _load_model_sync(self):
        """Load model synchronously (called from thread pool)."""
        model_path = self._get_model_path()
        
        # Initialize config
        config = ExLlamaV2Config()
        config.model_dir = str(model_path)
        config.prepare()
        
        # Adjust context length based on available VRAM
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb >= 28:
            config.max_seq_len = 8192
        else:
            config.max_seq_len = 4096
        
        logger.info(f"ExLlamaV2 config: max_seq_len={config.max_seq_len}, GPU={gpu_memory_gb:.1f}GB")
        
        # Load model
        self.model = ExLlamaV2(config)
        self.model.load()
        
        # Load tokenizer
        self.tokenizer = ExLlamaV2Tokenizer(config)
        
        # Create cache for KV storage - must NOT be lazy for streaming generation
        # max_seq_len must match config for proper cache allocation
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=config.max_seq_len, lazy=False)
        
        # Create streaming generator
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        
        # Store config for reference
        self._config = config
        
        # Extract stop token IDs from tokenizer (the proper way)
        self._stop_token_ids = self._build_stop_token_ids()
        
        logger.info(f"ExLlamaV2 model loaded: {self.model_id}")
    
    def _build_stop_token_ids(self) -> List[int]:
        """
        Extract stop token IDs from the tokenizer.
        
        This is the industry-standard approach: use actual token IDs, not strings.
        For Llama 3 models, we need:
          - EOS token (end of sequence)
          - <|eot_id|> (end of turn - ID 128009 in Llama 3)
          - <|start_header_id|> (start of new turn header - ID 128006)
        """
        stop_ids = []
        
        # 1. Always include EOS token
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
            logger.debug(f"Added EOS token ID: {self.tokenizer.eos_token_id}")
        
        # 2. Llama 3 specific stop tokens - encode them to get actual IDs
        llama3_stop_tokens = [
            "<|eot_id|>",           # End of turn (most important)
            "<|start_header_id|>",  # Start of new header = new turn starting
            "<|end_of_text|>",      # End of text
        ]
        
        for token_str in llama3_stop_tokens:
            try:
                # Encode the special token - ExLlamaV2 returns a tensor
                ids = self.tokenizer.encode(token_str, add_bos=False, add_eos=False)
                if ids is not None and ids.numel() == 1:
                    token_id = ids.item()
                    if token_id not in stop_ids:
                        stop_ids.append(token_id)
                        logger.debug(f"Added stop token '{token_str}' -> ID {token_id}")
            except Exception as e:
                logger.debug(f"Could not encode stop token '{token_str}': {e}")
        
        # 3. Fallback: Llama 3 known token IDs if encoding failed
        # These are standard for Llama 3 Instruct models
        LLAMA3_KNOWN_STOP_IDS = [128009, 128006]  # eot_id, start_header_id
        for known_id in LLAMA3_KNOWN_STOP_IDS:
            if known_id not in stop_ids:
                stop_ids.append(known_id)
                logger.debug(f"Added fallback Llama 3 stop ID: {known_id}")
        
        logger.info(f"Stop token IDs configured: {stop_ids}")
        return stop_ids
    
    async def generate(
        self, 
        prompt: str, 
        max_length: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95,
        language: str = "auto",
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate text response asynchronously with streaming."""
        logger.info(f"ExLlamaModel.generate called with prompt='{prompt[:50]}...', max_length={max_length}")
        
        if not self._loaded:
            logger.info("Model not loaded, attempting to load...")
            success = await self.load()
            if not success:
                yield "Error: Failed to load ExLlamaV2 model"
                return
        
        try:
            # Build the full prompt with chat template
            full_prompt = self._build_chat_prompt(prompt, system_prompt, conversation_history)
            logger.info(f"Built prompt with {len(full_prompt)} characters")
            
            # Create sampler settings
            settings = ExLlamaV2Sampler.Settings()
            settings.temperature = temperature
            settings.top_p = top_p
            settings.top_k = 50
            settings.token_repetition_penalty = 1.1
            
            # Calculate max tokens
            input_ids = self.tokenizer.encode(full_prompt)
            input_length = input_ids.shape[-1]
            max_new_tokens = min(max_length, self.max_context_tokens - input_length - 100)
            max_new_tokens = max(1, min(max_new_tokens, 2048))
            
            logger.info(f"Input tokens: {input_length}, max_new_tokens: {max_new_tokens}")
            
            # Set stop conditions using pre-computed token IDs (not strings)
            # This is the proper way - token IDs are reliable, strings are not
            self.generator.set_stop_conditions(self._stop_token_ids)
            logger.debug(f"Stop conditions set: {self._stop_token_ids}")
            
            self.generator.begin_stream(input_ids, settings)
            
            generated_text = ""
            chunk_count = 0
            
            # Stream generation
            while True:
                # Run generation step in thread pool
                result = await asyncio.to_thread(self._generate_step)
                
                if result is None:
                    break
                    
                chunk, eos = result
                
                if chunk:
                    chunk_count += 1
                    generated_text += chunk
                    logger.debug(f"Generated chunk #{chunk_count}: {repr(chunk)}")
                    yield chunk
                
                if eos:
                    logger.debug("EOS signal received from generator")
                    break
                
                # Safety limit on max tokens
                if chunk_count >= max_new_tokens:
                    logger.info(f"Reached max tokens limit: {chunk_count}")
                    break
                
                # Small delay for cooperative multitasking
                await asyncio.sleep(0.001)
            
            logger.info(f"ExLlamaModel generation complete. Total chunks: {chunk_count}")
            
        except Exception as e:
            logger.error(f"Error in ExLlamaModel.generate: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"Error: {str(e)}"
    
    def _generate_step(self):
        """Single generation step (called from thread pool)."""
        try:
            chunk, eos, _ = self.generator.stream()
            return (chunk, eos)
        except StopIteration:
            return None
        except Exception as e:
            logger.error(f"Error in generation step: {e}")
            return None
    
    def _build_chat_prompt(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build a chat-formatted prompt for Llama models."""
        # Default system prompt
        default_system = (
            "You are a helpful, harmless, and honest AI assistant. "
            "Respond directly and helpfully to the user's questions."
        )
        sys_text = system_prompt.strip() if system_prompt else default_system
        
        # Build Llama 3 chat format
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        # {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
        # {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        messages = []
        messages.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_text}<|eot_id|>")
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role == "user" and content:
                    messages.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == "assistant" and content:
                    messages.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        # Add current user message and assistant prefix
        messages.append(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt.strip()}<|eot_id|>")
        messages.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(messages)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "model_id": self.model_id,
            "loaded": self._loaded,
            "device": self.device,
            "quantization": "exl2",
            "supported_languages": list(self.language_codes.keys()),
            "language_names": self.language_codes,
            "cache_dir": self.cache_dir,
            "max_context_tokens": self.max_context_tokens,
            "conversation_memory": {
                "enabled": True,
                "max_context_tokens": self.max_context_tokens,
                "preferred_context_tokens": self.preferred_context_tokens,
                "min_context_tokens": self.min_context_tokens,
                "context_window_safety_margin": self.context_window_safety_margin,
                "max_history_messages": self.max_history_messages,
                "current_history_count": len(self.conversation_history),
                "current_token_count": self._context_token_count,
                "context_management": "token_based_optimization"
            }
        }
    
    async def unload(self):
        """Unload the model to free memory."""
        if self._loaded:
            # Clean up ExLlamaV2 objects
            if self.generator:
                del self.generator
                self.generator = None
            if self.cache:
                del self.cache
                self.cache = None
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            self._loaded = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded ExLlamaV2 model: {self.model_id}")
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        self._context_token_count = 0
        logger.info("Conversation history cleared")
    
    def add_to_conversation_history(self, role: str, content: str):
        """Add a message to conversation history."""
        if role in ["user", "assistant"] and content.strip():
            self.conversation_history.append({"role": role, "content": content.strip()})
            # Rough token estimation for ExLlama
            self._context_token_count += len(content) // 4
            logger.info(f"Added {role} message to history. Total messages: {len(self.conversation_history)}")
            
            # Clean up old messages if history gets too long
            while len(self.conversation_history) > self.max_history_messages:
                removed_msg = self.conversation_history.pop(0)
                self._context_token_count -= len(removed_msg["content"]) // 4
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get a copy of the conversation history."""
        return self.conversation_history.copy()
    
    def __del__(self):
        """Cleanup on deletion."""
        # Synchronous cleanup for destructor
        if hasattr(self, 'generator') and self.generator:
            del self.generator
        if hasattr(self, 'cache') and self.cache:
            del self.cache
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer

