"""
Qwen Model Integration

Handles Qwen model loading, inference, and management with multi-language support.
"""

import os
import contextlib
import torch
from typing import Dict, List, Optional, AsyncGenerator, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from loguru import logger
import asyncio


class QwenModel:
    """Qwen model wrapper with async support and multi-language capabilities."""
    
    def __init__(self, model_id: str, cache_dir: str, device: str = "auto", precision_mode: Optional[str] = None):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self.precision_mode = precision_mode  # e.g., "fp16" to force no quantization on capable GPUs
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
        # Configure quantization based on available VRAM and model size
        quantization_config = None
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Heuristic: infer model size class from model_id
            model_id_lower = self.model_id.lower()
            assumed_size_gb = 7
            if "14b" in model_id_lower:
                assumed_size_gb = 28
            elif "32b" in model_id_lower:
                assumed_size_gb = 64
            elif "7b" in model_id_lower:
                assumed_size_gb = 10
            elif "4b" in model_id_lower or "3b" in model_id_lower:
                assumed_size_gb = 6

            # If user explicitly requests FP16, try to avoid quantization when GPU is large enough
            if (self.precision_mode == "fp16") and gpu_memory_gb >= max(assumed_size_gb, 24):
                quantization_config = None
                logger.info(f"FP16 override enabled: loading {self.model_id} without quantization on {gpu_memory_gb:.1f}GB GPU")
            # Selection matrix
            elif assumed_size_gb <= 10 and gpu_memory_gb >= 12:
                quantization_config = None
                logger.info(f"Using FP16 (no quant) for {assumed_size_gb}GB model on {gpu_memory_gb:.1f}GB GPU")
            elif assumed_size_gb <= 28 and gpu_memory_gb >= 16:
                # 14B: prefer 8-bit on >=16GB
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                logger.info(f"Using 8-bit quantization for ~14B on {gpu_memory_gb:.1f}GB GPU")
            elif assumed_size_gb <= 64 and gpu_memory_gb >= 24:
                # 32B: use 4-bit to squeeze
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info(f"Using 4-bit quantization for ~32B on {gpu_memory_gb:.1f}GB GPU")
            else:
                # Fallback conservative 4-bit
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info(f"Fallback to 4-bit quantization (model ~{assumed_size_gb}GB, GPU {gpu_memory_gb:.1f}GB)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Load model with fallback if bitsandbytes is missing
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

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
        except Exception as e:
            error_message = str(e)
            # If quantization requested but bitsandbytes is missing, fall back to FP16 when VRAM is sufficient
            if "bitsandbytes" in error_message.lower() and self.device == "cuda" and torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Try FP16 fallback for big GPUs (>= 28GB), which can handle 14B in memory
                if gpu_memory_gb >= 28:
                    logger.warning("bitsandbytes not available; retrying load without quantization (FP16) on large GPU")
                    safe_kwargs = dict(model_kwargs)
                    safe_kwargs.pop("quantization_config", None)
                    safe_kwargs["device_map"] = "auto"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        **safe_kwargs
                    )
                else:
                    raise
            else:
                raise
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate(
        self, 
        prompt: str, 
        max_length: int = 4096,  # Increased default for better responses
        temperature: float = 0.8,  # Slightly higher for more natural responses
        top_p: float = 0.95,  # Higher top_p for better diversity
        language: str = "auto",
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
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
            # Build chat-style inputs using tokenizer chat template when available
            logger.info("Preparing inputs with chat template (if available)...")
            messages = []
            
            # Add a concise system guide to avoid meta/emoji artifacts
            default_system = (
                "You are a helpful assistant. Respond directly and succinctly. "
                "Do not include meta commentary, internal notes, or emojis unless explicitly asked."
            )
            sys_text = (system_prompt.strip() + "\n\n" + default_system) if system_prompt else default_system
            messages.append({"role": "system", "content": sys_text})
            
            # Add conversation history if provided
            if conversation_history:
                logger.info(f"Adding {len(conversation_history)} messages from conversation history")
                for i, msg in enumerate(conversation_history):
                    if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                        messages.append({"role": msg["role"], "content": msg["content"].strip()})
                        logger.info(f"  History message {i+1}: {msg['role']} - {msg['content'][:50]}...")
            else:
                logger.info("No conversation history provided")
            
            # Add current user message
            messages.append({"role": "user", "content": prompt.strip()})
            
            # Manage context window size
            messages = self._manage_context_window(messages)
            logger.info(f"Final message count after context management: {len(messages)}")

            if hasattr(self.tokenizer, "apply_chat_template") or getattr(self.tokenizer, "chat_template", None):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.info("Tokenizing chat-formatted text...")
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
            else:
                # Fallback: simple instruction + user text without role prefixes
                fallback = (
                    sys_text + "\n\n" + prompt.strip()
                )
                logger.info("Tokenizing fallback-formatted text...")
                inputs = self.tokenizer(
                    fallback,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
            logger.info(f"Input tokenized. Shape: {inputs['input_ids'].shape}")
            
            # Move inputs to the same device as the model's first layer
            if self.device == "cuda" and torch.cuda.is_available():
                logger.info("Moving inputs to model device...")
                # When using device_map="auto", get the device of the first parameter
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                logger.info(f"Inputs moved to {model_device}")
            
            # Generate response with real-time streaming
            logger.info("Starting streaming generation...")
            chunk_count = 0
            try:
                while True:
                    try:
                        # Use default to prevent StopIteration bubbling into Future
                        chunk = await asyncio.wait_for(
                            asyncio.to_thread(lambda: next(streamer, None)),
                            timeout=15.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Streamer timeout; stopping stream")
                        break
                    if chunk is None:
                        break
                    if not chunk:
                        continue
                    chunk_count += 1
                    logger.info(f"QwenModel yielding chunk #{chunk_count}: {repr(chunk)}")
                    yield chunk
            finally:
                # Ensure background task finishes
                with contextlib.suppress(Exception):
                    await task

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
            
            # Tighter token limits to prevent long, repetitive rambling
            if max_length <= 50:
                max_tokens_limit = 40
            elif max_length <= 100:
                max_tokens_limit = 80
            elif max_length <= 500:
                max_tokens_limit = 180
            else:
                max_tokens_limit = 200
                
            max_new_tokens = max(1, min(max_length - input_length, max_tokens_limit))
            logger.info(f"Input length: {input_length}, max_new_tokens: {max_new_tokens}, limit: {max_tokens_limit}")
            
            logger.info("Starting streaming generation token by token...")
            
            # Initialize for token-by-token generation
            current_input_ids = inputs['input_ids']
            generated_tokens = 0
            chunk_count = 0
            generated_text = ""
            
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
                    
                    if token_text.strip():
                        chunk_count += 1
                        generated_text += token_text
                        
                        # Check word count for early stopping
                        word_count = len(generated_text.split())
                        logger.info(f"Generated token #{chunk_count}: {repr(token_text)} (words: {word_count})")
                        
                        yield token_text
                        
                        # Improved stopping conditions
                        # Stop early for short requests
                        if max_length <= 30 and word_count >= 20:
                            logger.info(f"Stopping early: reached {word_count} words for very short response")
                            break
                        if max_length <= 50 and word_count >= 35:
                            logger.info(f"Stopping early: reached {word_count} words for short response")
                            break
                        # Global safety word cap
                        if word_count >= 220:
                            logger.info(f"Stopping at global word cap: {word_count} words")
                            break
                        
                        # Stop at sentence end once response is reasonably long
                        if token_text.strip().endswith(('.', '!', '?', '。', '！', '？')) and word_count >= 120:
                            logger.info(f"Stopping at sentence end after sufficient content: {word_count} words")
                            break
                        
                        # Repetition detection
                        if len(generated_text) > 200 and word_count > 60:
                            # n-gram style repetition using last 48 chars
                            recent_text = generated_text[-48:]
                            earlier_text = generated_text[:-48]
                            if recent_text.strip() and recent_text in earlier_text:
                                logger.warning("Repetition detected (recent segment already occurred), stopping")
                                break
                            # Common boilerplate phrases repetition (CN/EN)
                            repetitive_phrases = [
                                "如果您有其他问题",
                                "欢迎随时提问",
                                "祝您",
                                "请理解",
                                "如果还有其他问题",
                                "If you have any other questions",
                                "feel free to ask",
                            ]
                            repeated_hits = sum(generated_text.count(p) for p in repetitive_phrases)
                            if repeated_hits >= 2:
                                logger.warning("Detected multiple boilerplate repetitions, stopping")
                                break
                        
                        # Small delay for human observation
                        import asyncio
                        await asyncio.sleep(0.05)  # 50ms delay between tokens
                    
                    # Update input for next iteration
                    # Ensure next_token is on the same device as current_input_ids
                    next_token_device = next_token.to(current_input_ids.device)
                    current_input_ids = torch.cat([current_input_ids, next_token_device.unsqueeze(0)], dim=1)
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
    
    def _create_advanced_stopping_criteria(self):
        """Create advanced stopping criteria for better quality generation."""
        class AdvancedStoppingCriteria:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.generated_text = ""
                self.consecutive_newlines = 0
                self.max_consecutive_newlines = 2
                self.repetition_threshold = 3
                
            def __call__(self, input_ids, scores, **kwargs):
                # Decode the latest token
                latest_tokens = input_ids[0, -10:]  # Check last 10 tokens for context
                latest_text = self.tokenizer.decode(latest_tokens, skip_special_tokens=True)
                
                # Update generated text
                if len(latest_text) > len(self.generated_text):
                    new_text = latest_text[len(self.generated_text):]
                    self.generated_text += new_text
                    
                    # Check for excessive newlines
                    if new_text == '\n':
                        self.consecutive_newlines += 1
                        if self.consecutive_newlines >= self.max_consecutive_newlines:
                            return True  # Stop generation
                    else:
                        self.consecutive_newlines = 0
                    
                    # Check for repetitive patterns
                    if len(self.generated_text) > 100:
                        # Check for sentence-level repetition
                        sentences = self.generated_text.split('. ')
                        if len(sentences) >= 3:
                            recent_sentences = sentences[-2:]
                            for i, sentence in enumerate(sentences[:-2]):
                                if sentence.strip() in [s.strip() for s in recent_sentences]:
                                    return True  # Stop on repetition
                
                return False  # Continue generation
        
        return AdvancedStoppingCriteria(self.tokenizer)

    def _manage_context_window(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Manage context window size using token-based optimization."""
        # Calculate total tokens
        total_tokens = sum(self._count_tokens(msg.get("content", "")) for msg in messages)
        
        # If within limits, return as-is
        if total_tokens <= self.preferred_context_tokens:
            return messages
        
        logger.info(f"Context window management: {len(messages)} messages, {total_tokens} tokens, preferred: {self.preferred_context_tokens}")
        
        # Always keep system message and current user message
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        current_user_msg = messages[-1] if messages and messages[-1].get("role") == "user" else None
        
        # Get conversation messages (excluding system and current user)
        conversation_messages = messages[1:-1] if system_msg and current_user_msg else messages[:-1]
        
        # Calculate reserved tokens for system and current user
        reserved_tokens = 0
        if system_msg:
            reserved_tokens += self._count_tokens(system_msg.get("content", ""))
        if current_user_msg:
            reserved_tokens += self._count_tokens(current_user_msg.get("content", ""))
        
        # Target tokens for conversation history
        target_tokens = max(
            self.min_context_tokens - reserved_tokens,
            self.preferred_context_tokens - reserved_tokens - self.context_window_safety_margin
        )
        
        # Truncate conversation messages from the beginning, keeping most recent
        result = []
        if system_msg:
            result.append(system_msg)
        
        # Keep messages from most recent, fitting within token budget
        current_tokens = 0
        kept_messages = []
        
        for msg in reversed(conversation_messages):
            msg_tokens = self._count_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens <= target_tokens:
                kept_messages.insert(0, msg)  # Insert at beginning to maintain order
                current_tokens += msg_tokens
            else:
                break
        
        result.extend(kept_messages)
        
        if current_user_msg:
            result.append(current_user_msg)
        
        logger.info(f"Context optimized: {len(kept_messages)}/{len(conversation_messages)} conversation messages kept, "
                   f"{current_tokens + reserved_tokens} total tokens")
        
        return result
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if not self.tokenizer:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback estimation
            return len(text) // 4
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        self._context_token_count = 0
        logger.info("Conversation history cleared")
    
    def add_to_conversation_history(self, role: str, content: str):
        """Add a message to conversation history."""
        if role in ["user", "assistant"] and content.strip():
            self.conversation_history.append({"role": role, "content": content.strip()})
            self._context_token_count += self._count_tokens(content)
            logger.info(f"Added {role} message to history. Total messages: {len(self.conversation_history)}")
            
            # Clean up old messages if history gets too long
            while len(self.conversation_history) > self.max_history_messages:
                removed_msg = self.conversation_history.pop(0)
                self._context_token_count -= self._count_tokens(removed_msg["content"])
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get a copy of the conversation history."""
        return self.conversation_history.copy()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "model_id": self.model_id,
            "loaded": self._loaded,
            "device": self.device,
            "supported_languages": list(self.language_codes.keys()),
            "language_names": self.language_codes,
            "cache_dir": self.cache_dir,
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

