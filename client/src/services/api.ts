/**
 * API Client for Vibe LLM Backend
 * Handles all HTTP and WebSocket communication
 */

import type {
  ModelInfo,
  ChatRequest,
  ChatResponse,
  SystemInfo,
  HealthResponse,
  WebSocketMessage,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

class ApiClient {
  /**
   * Health check
   */
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
  }

  /**
   * Get all available models
   */
  async getModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${API_BASE}/models`);
    if (!response.ok) throw new Error('Failed to fetch models');
    const data = await response.json();
    return data.models || [];
  }

  /**
   * Load a specific model
   */
  async loadModel(modelName: string): Promise<{ success: boolean; error?: string }> {
    const response = await fetch(`${API_BASE}/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName }),
    });
    return response.json();
  }

  /**
   * Unload current model
   */
  async unloadModel(): Promise<{ success: boolean }> {
    const response = await fetch(`${API_BASE}/models/unload`, {
      method: 'POST',
    });
    return response.json();
  }

  /**
   * Get system information
   */
  async getSystemInfo(): Promise<SystemInfo> {
    const response = await fetch(`${API_BASE}/system/info`);
    if (!response.ok) throw new Error('Failed to fetch system info');
    return response.json();
  }

  /**
   * Simple chat (non-streaming)
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const params = new URLSearchParams({
      message: request.message,
      ...(request.model_name && { model_name: request.model_name }),
      ...(request.max_length && { max_length: request.max_length.toString() }),
      ...(request.temperature && { temperature: request.temperature.toString() }),
      ...(request.language && { language: request.language }),
    });

    const response = await fetch(`${API_BASE}/chat/simple?${params}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Chat request failed');
    }
    return response.json();
  }

  /**
   * Streaming chat via WebSocket
   */
  streamChat(
    message: string,
    onMessage: (msg: WebSocketMessage) => void,
    onError?: (error: Error) => void,
    onClose?: () => void,
    options?: {
      modelName?: string;
      maxLength?: number;
      temperature?: number;
      language?: string;
      conversationHistory?: Array<{role: string; content: string}>;
    }
  ): WebSocket {
    const ws = new WebSocket(`${WS_BASE}/ws`);

    ws.onopen = () => {
      const request = {
        type: 'chat',
        message,
        model_name: options?.modelName,
        max_length: options?.maxLength,
        temperature: options?.temperature,
        language: options?.language,
        conversation_history: options?.conversationHistory,
      };
      ws.send(JSON.stringify(request));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
        onError?.(error as Error);
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      onError?.(new Error('WebSocket connection error'));
    };

    ws.onclose = () => {
      onClose?.();
    };

    return ws;
  }

  /**
   * AI Detection via WebSocket
   */
  detectAI(
    text: string,
    onMessage: (msg: any) => void,
    onError?: (error: Error) => void,
    onClose?: () => void,
    options?: {
      detector?: string;
      useMultiple?: boolean;
    }
  ): WebSocket {
    const ws = new WebSocket(`${WS_BASE}/ws`);

    ws.onopen = () => {
      const request = {
        type: 'detect',
        text,
        detector: options?.detector,
        use_multiple: options?.useMultiple || false,
      };
      ws.send(JSON.stringify(request));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
        onError?.(error as Error);
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      onError?.(new Error('WebSocket connection error'));
    };

    ws.onclose = () => {
      onClose?.();
    };

    return ws;
  }
}

export const api = new ApiClient();

