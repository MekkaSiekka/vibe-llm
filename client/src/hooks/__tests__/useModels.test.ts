import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { useModels } from '../../hooks/useModels';
import { api } from '../../services/api';

vi.mock('../../services/api', () => {
  return {
    api: {
      getModels: vi.fn().mockResolvedValue([
        { name: 'Qwen3-0.6B', model_id: 'Qwen/Qwen3-0.6B', model_type: 'chat', size_gb: 1.2, device: 'cpu', available: true, loaded: false, recommended: false, languages: ['en', 'zh'] },
      ]),
      loadModel: vi.fn().mockResolvedValue({ success: true }),
      unloadModel: vi.fn().mockResolvedValue({ success: true }),
    },
  };
});

describe('useModels', () => {
  it('fetches and sets models', async () => {
    const { result } = renderHook(() => useModels());
    // Wait microtask queue
    await act(async () => {});
    expect(result.current.models.length).toBe(1);
  });

  it('loads a model', async () => {
    const { result } = renderHook(() => useModels());
    await act(async () => {});
    await act(async () => {
      await result.current.loadModel('Qwen3-0.6B');
    });
    expect(vi.mocked(api.loadModel)).toHaveBeenCalledWith('Qwen3-0.6B');
  });
});


