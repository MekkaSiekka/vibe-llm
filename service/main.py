"""
Main FastAPI Application

Provides REST API endpoints for local LLM interactions with Perplexity-like interface.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import uvicorn

from models.manager import ModelManager


# Global model manager instance
model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global model_manager
    
    # Startup
    logger.info("Starting Local LLM Service...")
    model_manager = ModelManager(cache_dir=os.getenv("MODEL_CACHE_DIR", "./models_cache"))
    
    # Auto-load best model
    result = await model_manager.auto_load_best_model()
    if "error" in result:
        logger.warning(f"Could not auto-load best model: {result['error']}")
    else:
        logger.info(f"Auto-loaded model: {result.get('model_name', 'unknown')}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local LLM Service...")
    if model_manager:
        await model_manager.unload_current_model()


# Create FastAPI app
app = FastAPI(
    title="Local LLM Service",
    description="Local LLM chat server with hot-switchable models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    model: Optional[str] = Field(None, description="Model name to use")
    max_length: int = Field(2048, description="Maximum response length")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling")
    language: str = Field("auto", description="Response language")
    stream: bool = Field(True, description="Stream response")


class ModelSwitchRequest(BaseModel):
    model_name: str = Field(..., description="Name of model to switch to")


class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    current_model: Optional[str] = None


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not model_manager:
        return HealthResponse(
            status="error",
            message="Model manager not initialized",
            model_loaded=False
        )
    
    current_model_info = await model_manager.get_current_model_info()
    current_model = current_model_info.get("model_id") if current_model_info else None
    
    return HealthResponse(
        status="healthy",
        message="Service is running",
        model_loaded=current_model_info is not None,
        current_model=current_model
    )


# System information endpoint
@app.get("/system")
async def get_system_info():
    """Get system and model information."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return await model_manager.get_system_info()


# List available models
@app.get("/models")
async def list_models():
    """Get all available models."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return await model_manager.get_all_available_models()


# Get model availability
@app.get("/models/{model_name}/availability")
async def get_model_availability(model_name: str):
    """Get availability status for a specific model."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    result = await model_manager.get_model_availability(model_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# Load/switch model
@app.post("/models/load")
async def load_model(request: ModelSwitchRequest):
    """Load or switch to a specific model."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    result = await model_manager.load_model(request.model_name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


# Unload current model
@app.post("/models/unload")
async def unload_model():
    """Unload the currently loaded model."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return await model_manager.unload_current_model()


# Chat endpoint (streaming)
@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """Chat with the LLM (streaming response)."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    # Switch model if specified
    if request.model:
        switch_result = await model_manager.load_model(request.model)
        if "error" in switch_result:
            raise HTTPException(status_code=400, detail=switch_result["error"])
    
    # Format messages into prompt
    prompt = _format_messages_to_prompt(request.messages)
    
    if request.stream:
        # Streaming response
        async def generate():
            try:
                async for chunk in model_manager.generate_response(
                    prompt=prompt,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    language=request.language
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in chat stream: {e}")
                yield f"data: Error: {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        # Non-streaming response
        response_chunks = []
        async for chunk in model_manager.generate_response(
            prompt=prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            language=request.language
        ):
            response_chunks.append(chunk)
        
        return {"response": "".join(response_chunks)}


# Simple chat endpoint
@app.post("/chat/simple")
async def simple_chat(message: str, model: Optional[str] = None):
    """Simple chat endpoint for quick testing."""
    logger.info(f"Simple chat request received: message='{message}', model={model}")
    
    if not model_manager:
        logger.error("Model manager not initialized")
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    # Switch model if specified
    if model:
        logger.info(f"Switching to model: {model}")
        switch_result = await model_manager.load_model(model)
        if "error" in switch_result:
            logger.error(f"Model switch failed: {switch_result['error']}")
            raise HTTPException(status_code=400, detail=switch_result["error"])
    
    logger.info(f"Starting generation for prompt: '{message}'")
    
    # Generate response
    response_chunks = []
    chunk_count = 0
    try:
        async for chunk in model_manager.generate_response(prompt=message):
            chunk_count += 1
            logger.info(f"Received chunk #{chunk_count}: {repr(chunk)}")
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        logger.info(f"Generation complete. Total chunks: {chunk_count}, Full response: {repr(full_response)}")
        
        return {"response": full_response}
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str = "default"):
    """WebSocket endpoint for real-time chat."""
    from service.websocket import handle_websocket
    await handle_websocket(websocket, client_id)


def _format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into a single prompt."""
    prompt_parts = []
    
    for message in messages:
        if message.role == "system":
            prompt_parts.append(f"System: {message.content}")
        elif message.role == "user":
            prompt_parts.append(f"Human: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}")
    
    return "\n\n".join(prompt_parts)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Local LLM Service",
        "version": "1.0.0",
        "description": "Local LLM chat server with hot-switchable models",
        "endpoints": {
            "health": "/health",
            "system": "/system",
            "models": "/models",
            "chat": "/chat",
            "websocket": "/ws"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable auto-reload to prevent shutdown during generation
        log_level="info"
    )

