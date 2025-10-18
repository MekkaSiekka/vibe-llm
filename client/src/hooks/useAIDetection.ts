/**
 * Custom hook for AI text detection functionality
 */

import { useState, useCallback, useRef } from 'react';
import { api } from '../services/api';

export interface DetectionResult {
  isAIGenerated: boolean;
  confidence: number;
  aiProbability: number;
  humanProbability: number;
  model: string;
  textLength: number;
  chunksProcessed: number;
  detectionMethod: string;
  processingTime: number;
}

export function useAIDetection() {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const analyzeText = useCallback((text: string, detector?: string) => {
    if (!text.trim()) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    wsRef.current = api.detectAI(
      text,
      (msg) => {
        console.log('Detection message received:', msg);

        if (msg.type === 'detection_start') {
          // Detection started
          console.log('Detection started');
        } else if (msg.type === 'detection_result') {
          // Detection completed successfully
          setResult({
            isAIGenerated: msg.is_ai_generated,
            confidence: msg.confidence,
            aiProbability: msg.ai_probability,
            humanProbability: msg.human_probability,
            model: msg.model,
            textLength: msg.text_length,
            chunksProcessed: msg.chunks_processed,
            detectionMethod: msg.detection_method,
            processingTime: msg.processing_time,
          });
          setIsAnalyzing(false);
        } else if (msg.type === 'detection_error') {
          // Detection failed
          setError(msg.content || 'Detection failed');
          setIsAnalyzing(false);
        }
      },
      (err) => {
        console.error('Detection WebSocket error:', err);
        setError(err.message || 'Connection error');
        setIsAnalyzing(false);
      },
      () => {
        console.log('Detection WebSocket closed');
        setIsAnalyzing(false);
      },
      {
        detector,
        useMultiple: false,
      }
    );
  }, []);

  const stopAnalysis = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsAnalyzing(false);
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return {
    result,
    isAnalyzing,
    error,
    analyzeText,
    stopAnalysis,
    clearResult,
  };
}

