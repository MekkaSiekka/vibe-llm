"""
WebSocket handler for real-time streaming chat.
"""
import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from models.manager import ModelManager


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                # Check if WebSocket is still open
                if websocket.client_state.name != "CONNECTED":
                    logger.warning(f"WebSocket {client_id} is not connected, removing")
                    self.disconnect(client_id)
                    return False
                    
                await websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False


# Global connection manager
manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket, client_id: str = "default"):
    """Handle WebSocket connections for real-time chat."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"WebSocket received message: {message_data}")
            
            # Extract message content
            message = message_data.get("message", "")
            model_id = message_data.get("model", None)
            max_length = message_data.get("max_length", 2048)
            temperature = message_data.get("temperature", 0.7)
            top_p = message_data.get("top_p", 0.9)
            
            if not message:
                await manager.send_message(client_id, {
                    "type": "error",
                    "content": "No message provided"
                })
                continue
            
            # Send acknowledgment
            await manager.send_message(client_id, {
                "type": "start",
                "content": "Starting generation..."
            })
            
            # Get model manager from global state
            from service.main import model_manager
            if not model_manager:
                await manager.send_message(client_id, {
                    "type": "error",
                    "content": "Model manager not available"
                })
                continue
            
            # Switch model if requested
            if model_id:
                logger.info(f"Switching to model: {model_id}")
                switch_result = await model_manager.load_model(model_id)
                if "error" in switch_result:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "content": f"Failed to switch model: {switch_result['error']}"
                    })
                    continue
            
            # Stream response
            try:
                chunk_count = 0
                full_response = ""
                
                async for chunk in model_manager.generate_response(
                    prompt=message,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                ):
                    chunk_count += 1
                    full_response += chunk
                    
                    # Send chunk to client
                    success = await manager.send_message(client_id, {
                        "type": "chunk",
                        "content": chunk,
                        "chunk_id": chunk_count
                    })
                    
                    # If sending failed, client disconnected
                    if not success:
                        logger.warning(f"Client {client_id} disconnected during streaming")
                        break
                
                # Send completion message
                await manager.send_message(client_id, {
                    "type": "complete",
                    "content": "Generation complete",
                    "total_chunks": chunk_count,
                    "full_response": full_response
                })
                
                logger.info(f"WebSocket generation complete for {client_id}: {chunk_count} chunks")
                
            except Exception as e:
                logger.error(f"Error during WebSocket generation: {e}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "content": f"Generation error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)