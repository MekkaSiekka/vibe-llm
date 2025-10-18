/**
 * Custom hook for managing model state
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type { ModelInfo } from '../types';

export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getModels();
      setModels(data);
      
      // Find currently loaded model (prefer the one we just loaded, or any loaded model)
      // Note: backend can have both a chat model and detector loaded simultaneously
      const loaded = data.find(m => m.loaded && m.name === currentModel) 
                  || data.find(m => m.loaded);
      setCurrentModel(loaded?.name || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models');
    } finally {
      setLoading(false);
    }
  }, [currentModel]);

  const loadModel = useCallback(async (modelName: string) => {
    try {
      setLoading(true);
      setError(null);
      const result = await api.loadModel(modelName);
      
      if (result.success) {
        setCurrentModel(modelName);
        await fetchModels(); // Refresh model list
      } else {
        setError(result.error || 'Failed to load model');
      }
      
      return result;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to load model';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setLoading(false);
    }
  }, [fetchModels]);

  const unloadModel = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      await api.unloadModel();
      setCurrentModel(null);
      await fetchModels(); // Refresh model list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unload model');
    } finally {
      setLoading(false);
    }
  }, [fetchModels]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return {
    models,
    currentModel,
    loading,
    error,
    loadModel,
    unloadModel,
    refresh: fetchModels,
  };
}

