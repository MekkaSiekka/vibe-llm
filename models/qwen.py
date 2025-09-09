"""
Qwen Model Integration

Handles Qwen model loading, inference, and management with multi-language support.
"""

import os
import torch
from typing import Dict, List, Optional, AsyncGenerator, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger
import asyncio


class QwenModel:
    """Qwen model wrapper with async support and multi-language capabilities."""
    
    def __init__(self, model_id: str, cache_dir: str, device: str = "auto"):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        # Language mappings for Qwen
        self.language_codes = {
            "en": "English",
            "zh": "中文",
            "fr": "Français", 
            "de": "Deutsch",
            "es": "Español",
            "ru": "Русский",
            "ja": "日本語",
            "ko": "한국어"
        }
    
    async def load(self) -> bool:
        """Load the Qwen model asynchronously."""
        if self._loaded:
            return True
            
        try:
            logger.info(f"Loading Qwen model: {self.model_id}")
            
            # Load model directly for fastest performance
            self._load_model()
            
            self._loaded = True
            logger.info(f"Successfully loaded Qwen model: {self.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model {self.model_id}: {e}")
            return False
    
    def _load_model(self):
        """Load model synchronously in thread pool."""
        # Configure quantization based on available VRAM
        quantization_config = None
        if self.device == "cuda" and torch.cuda.is_available():
            # Check available VRAM to decide quantization level
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory_gb >= 12:
                # High-end GPU: Use FP16 for best quality
                quantization_config = None
                logger.info(f"Using FP16 precision for high-end GPU ({gpu_memory_gb:.1f}GB VRAM)")
            elif gpu_memory_gb >= 8:
                # Mid-range GPU: Use 8-bit quantization for balance
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                logger.info(f"Using 8-bit quantization for mid-range GPU ({gpu_memory_gb:.1f}GB VRAM)")
            else:
                # Low VRAM: Use 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info(f"Using 4-bit quantization for low VRAM GPU ({gpu_memory_gb:.1f}GB VRAM)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Load model
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate(
        self, 
        prompt: str, 
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        language: str = "auto"
    ) -> AsyncGenerator[str, None]:
        """Generate text response asynchronously."""
        logger.info(f"QwenModel.generate called with prompt='{prompt}', max_length={max_length}")
        
        if not self._loaded:
            logger.info("Model not loaded, attempting to load...")
            await self.load()
        
        if not self._loaded:
            logger.error("Model failed to load")
            yield "Error: Model not loaded"
            return
        
        logger.info("Model is loaded, starting generation...")
        
        try:
            # Add language-specific system prompt if needed
            formatted_prompt = self._format_prompt(prompt, language)
            logger.info(f"Formatted prompt: {repr(formatted_prompt)}")
            
            # Tokenize input
            logger.info("Tokenizing input...")
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            logger.info(f"Input tokenized. Shape: {inputs['input_ids'].shape}")
            
            if self.device == "cuda" and torch.cuda.is_available():
                logger.info("Moving inputs to CUDA...")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                logger.info("Inputs moved to CUDA")
            
            # Generate response with real-time streaming
            logger.info("Starting streaming generation...")
            chunk_count = 0
            try:
                async for chunk in self._generate_async(
                    inputs,
                    max_length,
                    temperature,
                    top_p
                ):
                    chunk_count += 1
                    logger.info(f"QwenModel yielding chunk #{chunk_count}: {repr(chunk)}")
                    yield chunk
                    
                logger.info("Streaming generation completed")
            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                raise
            
            logger.info(f"QwenModel streaming complete. Total chunks: {chunk_count}")
                
        except Exception as e:
            logger.error(f"Error in QwenModel.generate: {e}")
            import traceback
            logger.error(f"QwenModel traceback: {traceback.format_exc()}")
            yield f"Error: {str(e)}"
    
    async def _generate_async(self, inputs, max_length, temperature, top_p):
        """Generate response asynchronously with real-time streaming."""
        logger.info(f"_generate_async called with max_length={max_length}, temperature={temperature}, top_p={top_p}")
        
        with torch.no_grad():
            # Calculate max_new_tokens safely with better limits
            input_length = inputs['input_ids'].shape[1]
            
            # More reasonable token limits to prevent runaway generation
            if max_length <= 50:
                max_tokens_limit = 40
            elif max_length <= 100:
                max_tokens_limit = 80
            elif max_length <= 500:
                max_tokens_limit = 250
            else:
                max_tokens_limit = 500  # Hard safety limit
                
            max_new_tokens = max(1, min(max_length - input_length, max_tokens_limit))
            logger.info(f"Input length: {input_length}, max_new_tokens: {max_new_tokens}, limit: {max_tokens_limit}")
            
            logger.info("Starting streaming generation token by token...")
            
            # Initialize for token-by-token generation
            current_input_ids = inputs['input_ids']
            generated_tokens = 0
            chunk_count = 0
            generated_text = ""  # Track full generated text for word counting
            
            # Safety timeout - prevent infinite loops
            import time
            start_time = time.time()
            max_generation_time = min(120, max_new_tokens * 0.3)  # Max 120 seconds or 0.3s per token
            
            # Generate tokens one by one for true streaming
            while generated_tokens < max_new_tokens:
                # Safety timeout check
                elapsed_time = time.time() - start_time
                if elapsed_time > max_generation_time:
                    logger.warning(f"Generation timeout after {elapsed_time:.2f}s, stopping")
                    break
                try:
                    # Generate next token
                    with torch.no_grad():
                        outputs = self.model(input_ids=current_input_ids)
                        logits = outputs.logits[0, -1, :]  # Get logits for last position
                    
                    # Apply temperature and top_p sampling
                    if temperature > 0:
                        logits = logits / temperature
                        
                        # Top-p (nucleus) sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                            sorted_indices_to_remove[0] = 0
                            
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            logits[indices_to_remove] = -float('inf')
                        
                        # Sample from the filtered distribution
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy sampling
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Enhanced EOS token detection
                    token_id = next_token.item()
                    if (token_id == self.tokenizer.eos_token_id or 
                        (hasattr(self.tokenizer, 'pad_token_id') and token_id == self.tokenizer.pad_token_id) or
                        token_id == self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else False):
                        logger.info(f"EOS/PAD/SEP token detected (id: {token_id}), stopping generation")
                        break
                    
                    # Decode the new token
                    token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    
                    if token_text.strip():  # Only yield non-empty tokens
                        chunk_count += 1
                        generated_text += token_text
                        
                        # Check word count for early stopping
                        word_count = len(generated_text.split())
                        logger.info(f"Generated token #{chunk_count}: {repr(token_text)} (words: {word_count})")
                        
                        yield token_text
                        
                        # More reasonable stopping conditions
                        # Only stop early for very short requests
                        if max_length <= 30 and word_count >= 20:  # Very short responses only
                            logger.info(f"Stopping early: reached {word_count} words for very short response")
                            break
                        elif max_length <= 50 and word_count >= 35:  # Short responses
                            logger.info(f"Stopping early: reached {word_count} words for short response")
                            break
                        elif word_count >= 300:  # Safety limit for all responses
                            logger.info(f"Stopping at safety word limit: {word_count} words")
                            break
                        
                        # Stop at natural sentence endings only for short requests
                        if (max_length <= 50 and word_count >= 15 and 
                            token_text.strip().endswith(('.', '!', '?', '。', '！', '？'))):
                            logger.info(f"Stopping at sentence end: {word_count} words")
                            break
                        
                        # Repetition detection - check if we're repeating recent content
                        if len(generated_text) > 150 and word_count > 50:  # Only check after substantial content
                            recent_text = generated_text[-30:]  # Last 30 characters
                            earlier_text = generated_text[:-30]  # Everything before that
                            # Only stop if we have significant repetition AND reasonable length
                            if recent_text in earlier_text and len(recent_text.strip()) > 10:
                                logger.warning(f"Significant repetition detected, stopping generation at {word_count} words")
                                break
                        
                        # Small delay for human observation
                        import asyncio
                        await asyncio.sleep(0.05)  # 50ms delay between tokens
                    
                    # Update input for next iteration
                    current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=1)
                    generated_tokens += 1
                    
                except Exception as e:
                    logger.error(f"Error generating token {generated_tokens}: {e}")
                    break
            
            logger.info("_generate_async completed")
    
    def _format_prompt(self, prompt: str, language: str) -> str:
        """Format prompt with proper conversation context."""
        # Detect if the prompt is in Chinese and format accordingly
        if any('\u4e00' <= char <= '\u9fff' for char in prompt):
            # Chinese prompt - use Chinese conversation format
            if language == "auto":
                return f"用户: {prompt.strip()}\n\n助手: "
            elif language in self.language_codes:
                lang_name = self.language_codes[language]
                return f"用户: {prompt.strip()}\n请用{lang_name}回答。\n\n助手: "
        else:
            # English prompt - use English conversation format
            if language == "auto":
                return f"User: {prompt.strip()}\n\nAssistant: "
            elif language in self.language_codes:
                lang_name = self.language_codes[language]
                return f"User: {prompt.strip()}\nPlease respond in {lang_name}.\n\nAssistant: "
        
        # Fallback for other cases
        return f"User: {prompt.strip()}\n\nAssistant: "
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "model_id": self.model_id,
            "loaded": self._loaded,
            "device": self.device,
            "supported_languages": list(self.language_codes.keys()),
            "language_names": self.language_codes,
            "cache_dir": self.cache_dir
        }
    
    async def unload(self):
        """Unload the model to free memory."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {self.model_id}")
        
        # No thread pool to shutdown
    
    def __del__(self):
        """Cleanup on deletion."""
        pass  # No thread pool to cleanup

