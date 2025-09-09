#!/usr/bin/env python3
"""
Example client for Local LLM Service

Demonstrates how to interact with the service using both REST API and WebSocket.
"""

import asyncio
import httpx
import websockets
import json
from typing import AsyncGenerator


class LLMClient:
    """Client for interacting with the Local LLM Service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
    
    async def chat_rest(self, message: str, model: str = None) -> str:
        """Chat using REST API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/simple",
                json={"message": message, "model": model}
            )
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"REST API error: {response.status_code}")
    
    async def chat_stream_rest(self, message: str, model: str = None) -> AsyncGenerator[str, None]:
        """Chat using REST API with streaming."""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat",
                json={
                    "messages": [{"role": "user", "content": message}],
                    "model": model,
                    "stream": True
                }
            ) as response:
                if response.status_code == 200:
                    async for chunk in response.aiter_text():
                        if chunk.startswith("data: ") and not chunk.endswith("[DONE]\n\n"):
                            yield chunk[6:]  # Remove "data: " prefix
                else:
                    raise Exception(f"REST streaming error: {response.status_code}")
    
    async def chat_websocket(self, message: str, model: str = None) -> str:
        """Chat using WebSocket."""
        uri = f"{self.ws_url}/ws"
        
        async with websockets.connect(uri) as websocket:
            # Send chat message
            await websocket.send(json.dumps({
                "type": "chat",
                "data": {
                    "message": message,
                    "model": model
                }
            }))
            
            # Receive response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["success"]:
                return data["data"]["response"]
            else:
                raise Exception(f"WebSocket error: {data['error']}")
    
    async def get_models(self) -> list:
        """Get available models."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/models")
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Models API error: {response.status_code}")
    
    async def get_system_info(self) -> dict:
        """Get system information."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/system")
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"System info API error: {response.status_code}")


async def demo_rest_api():
    """Demonstrate REST API usage."""
    print("🌐 REST API Demo")
    print("=" * 50)
    
    client = LLMClient()
    
    try:
        # Get system info
        print("📊 System Information:")
        system_info = await client.get_system_info()
        print(f"   Platform: {system_info['hardware']['hardware']['platform']}")
        print(f"   Available models: {len(system_info['available_models'])}")
        print()
        
        # Get models
        print("🤖 Available Models:")
        models = await client.get_models()
        for model in models[:3]:
            print(f"   - {model['name']} ({model['size_gb']}GB)")
        print()
        
        # Simple chat
        print("💬 Simple Chat:")
        response = await client.chat_rest("Hello, how are you?")
        print(f"   Response: {response[:100]}...")
        print()
        
        # Streaming chat
        print("📡 Streaming Chat:")
        print("   Response: ", end="", flush=True)
        async for chunk in client.chat_stream_rest("What is 2+2?"):
            print(chunk, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"❌ REST API demo failed: {e}")


async def demo_websocket():
    """Demonstrate WebSocket usage."""
    print("🔌 WebSocket Demo")
    print("=" * 50)
    
    client = LLMClient()
    
    try:
        # WebSocket chat
        print("💬 WebSocket Chat:")
        response = await client.chat_websocket("Tell me a short joke")
        print(f"   Response: {response}")
        print()
        
    except Exception as e:
        print(f"❌ WebSocket demo failed: {e}")


async def interactive_chat():
    """Interactive chat session."""
    print("🎯 Interactive Chat")
    print("=" * 50)
    print("Type 'quit' to exit, 'models' to see available models")
    print()
    
    client = LLMClient()
    
    try:
        while True:
            message = input("You: ").strip()
            
            if message.lower() == 'quit':
                break
            elif message.lower() == 'models':
                models = await client.get_models()
                print("Available models:")
                for model in models:
                    print(f"  - {model['name']} ({model['size_gb']}GB)")
                print()
                continue
            elif not message:
                continue
            
            print("Assistant: ", end="", flush=True)
            try:
                async for chunk in client.chat_stream_rest(message):
                    print(chunk, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"Error: {e}\n")
                
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"❌ Interactive chat failed: {e}")


async def main():
    """Main demo function."""
    print("🚀 Local LLM Service Client Demo")
    print("=" * 60)
    print()
    
    # Run demos
    await demo_rest_api()
    await demo_websocket()
    
    # Interactive chat
    print("🎮 Starting Interactive Chat...")
    print("(Make sure the service is running: docker-compose up -d)")
    print()
    await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())

