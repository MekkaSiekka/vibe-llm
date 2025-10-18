/**
 * Unit Tests for useAIDetection Hook
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useAIDetection } from '../useAIDetection';
import { api } from '../../services/api';

// Mock the API
vi.mock('../../services/api', () => ({
  api: {
    detectAI: vi.fn(),
  },
}));

describe('useAIDetection', () => {
  let mockWebSocket: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Create mock WebSocket
    mockWebSocket = {
      close: vi.fn(),
      send: vi.fn(),
      onopen: null,
      onmessage: null,
      onerror: null,
      onclose: null,
    };

    (api.detectAI as any).mockImplementation((_text: string, onMessage: (msg: any) => void, onError?: (error: Error) => void, onClose?: () => void) => {
      mockWebSocket.onMessage = onMessage;
      mockWebSocket.onError = onError;
      mockWebSocket.onClose = onClose;
      return mockWebSocket;
    });
  });

  describe('Initial State', () => {
    it('starts with null result', () => {
      const { result } = renderHook(() => useAIDetection());
      
      expect(result.current.result).toBeNull();
      expect(result.current.isAnalyzing).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  describe('analyzeText', () => {
    it('starts analysis and sets analyzing state', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      expect(result.current.isAnalyzing).toBe(true);
      expect(api.detectAI).toHaveBeenCalledWith(
        'test text',
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
        { detector: undefined, useMultiple: false }
      );
    });

    it('ignores empty text', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('');
      });
      
      expect(result.current.isAnalyzing).toBe(false);
      expect(api.detectAI).not.toHaveBeenCalled();
    });

    it('ignores whitespace-only text', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('   ');
      });
      
      expect(result.current.isAnalyzing).toBe(false);
      expect(api.detectAI).not.toHaveBeenCalled();
    });

    it('passes detector name to API', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text', 'roberta-detector');
      });
      
      expect(api.detectAI).toHaveBeenCalledWith(
        'test text',
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
        { detector: 'roberta-detector', useMultiple: false }
      );
    });
  });

  describe('Result Handling', () => {
    it('handles detection_result message', async () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      const detectionResult = {
        type: 'detection_result',
        is_ai_generated: true,
        confidence: 0.95,
        ai_probability: 0.95,
        human_probability: 0.05,
        model: 'roberta-base',
        text_length: 9,
        chunks_processed: 1,
        detection_method: 'transformer_classification',
        processing_time: 1.5,
      };
      
      act(() => {
        mockWebSocket.onMessage(detectionResult);
      });
      
      await waitFor(() => {
        expect(result.current.result).toEqual({
          isAIGenerated: true,
          confidence: 0.95,
          aiProbability: 0.95,
          humanProbability: 0.05,
          model: 'roberta-base',
          textLength: 9,
          chunksProcessed: 1,
          detectionMethod: 'transformer_classification',
          processingTime: 1.5,
        });
        expect(result.current.isAnalyzing).toBe(false);
        expect(result.current.error).toBeNull();
      });
    });

    it('handles detection_error message', async () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      act(() => {
        mockWebSocket.onMessage({
          type: 'detection_error',
          content: 'Model not loaded',
        });
      });
      
      await waitFor(() => {
        expect(result.current.error).toBe('Model not loaded');
        expect(result.current.isAnalyzing).toBe(false);
        expect(result.current.result).toBeNull();
      });
    });

    it('handles detection_start message', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      act(() => {
        mockWebSocket.onMessage({
          type: 'detection_start',
        });
      });
      
      expect(result.current.isAnalyzing).toBe(true);
    });
  });

  describe('stopAnalysis', () => {
    it('closes WebSocket and stops analyzing', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      act(() => {
        result.current.stopAnalysis();
      });
      
      expect(mockWebSocket.close).toHaveBeenCalled();
      expect(result.current.isAnalyzing).toBe(false);
    });

    it('handles stop when no WebSocket exists', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.stopAnalysis();
      });
      
      expect(result.current.isAnalyzing).toBe(false);
    });
  });

  describe('clearResult', () => {
    it('clears result and error', async () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      act(() => {
        mockWebSocket.onMessage({
          type: 'detection_result',
          is_ai_generated: true,
          confidence: 0.95,
          ai_probability: 0.95,
          human_probability: 0.05,
          model: 'test',
          text_length: 9,
          chunks_processed: 1,
          detection_method: 'test',
          processing_time: 1.0,
        });
      });
      
      await waitFor(() => {
        expect(result.current.result).not.toBeNull();
      });
      
      act(() => {
        result.current.clearResult();
      });
      
      expect(result.current.result).toBeNull();
      expect(result.current.error).toBeNull();
    });
  });

  describe('Error Handling', () => {
    it('handles WebSocket errors', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      act(() => {
        mockWebSocket.onError(new Error('Connection failed'));
      });
      
      expect(result.current.error).toBe('Connection failed');
      expect(result.current.isAnalyzing).toBe(false);
    });

    it('handles WebSocket close', () => {
      const { result } = renderHook(() => useAIDetection());
      
      act(() => {
        result.current.analyzeText('test text');
      });
      
      act(() => {
        mockWebSocket.onClose();
      });
      
      expect(result.current.isAnalyzing).toBe(false);
    });
  });

  describe('Multiple Analyses', () => {
    it('can perform multiple analyses', async () => {
      const { result } = renderHook(() => useAIDetection());
      
      // First analysis
      act(() => {
        result.current.analyzeText('text 1');
      });
      
      act(() => {
        mockWebSocket.onMessage({
          type: 'detection_result',
          is_ai_generated: true,
          confidence: 0.9,
          ai_probability: 0.9,
          human_probability: 0.1,
          model: 'test',
          text_length: 6,
          chunks_processed: 1,
          detection_method: 'test',
          processing_time: 1.0,
        });
      });
      
      await waitFor(() => {
        expect(result.current.result?.confidence).toBe(0.9);
      });
      
      // Second analysis
      act(() => {
        result.current.analyzeText('text 2');
      });
      
      act(() => {
        mockWebSocket.onMessage({
          type: 'detection_result',
          is_ai_generated: false,
          confidence: 0.8,
          ai_probability: 0.2,
          human_probability: 0.8,
          model: 'test',
          text_length: 6,
          chunks_processed: 1,
          detection_method: 'test',
          processing_time: 1.0,
        });
      });
      
      await waitFor(() => {
        expect(result.current.result?.confidence).toBe(0.8);
        expect(result.current.result?.isAIGenerated).toBe(false);
      });
    });
  });
});

