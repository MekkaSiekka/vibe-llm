/**
 * API Types for Vibe LLM Backend
 */

export interface ModelInfo {
  name: string;
  model_id: string;
  model_type: 'chat' | 'ai_detector';
  size_gb: number;
  device: 'cpu' | 'cuda';
  available: boolean;
  loaded: boolean;
  recommended: boolean;
  languages: string[];
  description?: string;
  accuracy?: number; // For AI detection models
  mobile_optimized?: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export interface ChatRequest {
  message: string;
  model_name?: string;
  max_length?: number;
  temperature?: number;
  language?: string;
}

export interface ChatResponse {
  response: string;
  model_name: string;
  success: boolean;
  error?: string;
}

export interface SystemInfo {
  cpu: string;
  ram_gb: number;
  has_gpu: boolean;
  gpu_name?: string;
  gpu_memory_gb?: number;
  available_memory_gb: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name?: string;
  available_models: number;
}

export type Tab = 'chat' | 'ai-detect' | 'settings';

export interface WebSocketMessage {
  type: 'chunk' | 'done' | 'complete' | 'error';
  content: string;
  model_name?: string;
  total_chunks?: number;
  full_response?: string;
}

