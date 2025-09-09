# 🚀 Vibe LLM

A high-performance local LLM service with **real-time WebSocket streaming** and **hot-swappable models**. Built for speed, scalability, and that ChatGPT-like streaming experience.

## ✨ Features

- **🔥 Real-time WebSocket Streaming**: Word-by-word generation like ChatGPT
- **⚡ Hot-swappable Models**: Switch between models without restarting
- **🎯 Smart Length Control**: Respects word limits and stops appropriately
- **🚀 GPU Acceleration**: CUDA support with automatic hardware detection
- **📡 REST + WebSocket APIs**: Multiple ways to interact
- **🔧 Production Ready**: Comprehensive error handling and logging

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (optional but recommended)
- 8GB+ RAM (16GB+ recommended for larger models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vibe-llm.git
   cd vibe-llm
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # For faster downloads
   pip install hf_xet "huggingface_hub[hf_xet]"
   ```

4. **Start the service**
   ```bash
   python -m service.main
   ```

The service will start on `http://localhost:8000` with WebSocket at `ws://localhost:8000/ws`

## 🎮 Usage

### WebSocket Streaming (Recommended)
```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Send message
        message = {
            "message": "Tell me about quantum computing",
            "max_length": 100,
            "temperature": 0.7
        }
        await websocket.send(json.dumps(message))
        
        # Receive streaming response
        async for response in websocket:
            data = json.loads(response)
            if data["type"] == "chunk":
                print(data["content"], end="", flush=True)
            elif data["type"] == "complete":
                break

asyncio.run(chat())
```

### REST API
```bash
# Simple chat
curl -X POST "http://localhost:8000/chat/simple?message=Hello"

# Switch models
curl -X POST "http://localhost:8000/models/load" \
     -H "Content-Type: application/json" \
     -d '{"model_name": "Qwen2.5-7B-Instruct"}'

# List available models
curl "http://localhost:8000/models"
```

## 🤖 Supported Models

| Model | Size | VRAM | Performance | Best For |
|-------|------|------|-------------|----------|
| **Qwen3-0.6B** | 1.2GB | ~2GB | ⚡⚡⚡ | Quick responses, testing |
| **Qwen2.5-3B** | 6GB | ~4GB | ⚡⚡ | Balanced speed/quality |
| **Qwen2.5-7B** | 10GB | ~8GB | ⚡ | Best quality responses |

*Auto-detects your hardware and recommends optimal models*

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   Model Manager  │    │   Qwen Models   │
│   Streaming     │◄──►│   (Hot-swap)     │◄──►│   (GPU/CPU)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   REST API      │    │   Hardware       │    │   Token-by-     │
│   Endpoints     │◄──►│   Detection      │◄──►│   Token Gen     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables
```bash
MODEL_CACHE_DIR=./models_cache  # Model storage location
CUDA_VISIBLE_DEVICES=0          # GPU selection
```

### Model Configuration
Models are auto-detected based on your hardware in `models/detector.py`:
- **12GB+ VRAM**: All models available
- **8GB+ VRAM**: Up to 7B models
- **4GB+ VRAM**: Up to 3B models
- **CPU only**: Smaller models

## 🚀 Performance

- **Real-time streaming**: ~50ms between tokens for smooth UX
- **GPU acceleration**: 10x faster than CPU-only
- **Smart caching**: Models stay loaded for instant switching
- **Optimized inference**: Token-by-token generation with proper sampling

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test WebSocket streaming
python simple_websocket_test.py

# Test model loading
python test_7b_model.py

# Load larger model
python load_7b.py
```

## 📚 API Reference

### WebSocket Messages
```json
// Send
{
  "message": "Your prompt here",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "model": "Qwen2.5-7B-Instruct"  // optional
}

// Receive
{"type": "start", "content": "Starting generation..."}
{"type": "chunk", "content": "Hello", "chunk_id": 1}
{"type": "complete", "content": "Generation complete", "total_chunks": 50}
{"type": "error", "content": "Error message"}
```

### REST Endpoints
- `GET /health` - Service health check
- `GET /models` - List available models  
- `POST /models/load` - Load/switch model
- `POST /chat/simple` - Simple chat endpoint
- `POST /chat` - Full chat with streaming
- `GET /system` - System information

## 🛠️ Development

### Project Structure
```
vibe-llm/
├── models/           # Model implementations
│   ├── manager.py    # Model lifecycle management
│   ├── qwen.py      # Qwen model wrapper
│   └── detector.py   # Hardware detection
├── service/         # FastAPI service
│   ├── main.py      # Main application
│   └── websocket.py # WebSocket handlers
├── tests/           # Unit tests
└── requirements.txt # Dependencies
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📈 Roadmap

- [ ] Support for more model families (Llama, Mistral, etc.)
- [ ] Model quantization (4-bit, 8-bit)
- [ ] Multi-GPU support
- [ ] Docker deployment
- [ ] Web UI interface
- [ ] Conversation memory/context
- [ ] Custom fine-tuned model support

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent models
- **Hugging Face** for the transformers library
- **FastAPI** for the amazing web framework
- **The community** for testing and feedback

---

**Built with ❤️ for the local LLM community**

*Get that real-time AI vibe without the cloud! 🌟*