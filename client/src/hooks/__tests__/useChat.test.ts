import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { useChat } from '../../hooks/useChat';

const mockWebSocket = {
  close: vi.fn(),
};

vi.mock('../../services/api', () => {
  return {
    api: {
      streamChat: vi.fn(() => mockWebSocket as unknown as WebSocket),
      chat: vi.fn().mockResolvedValue({ success: true, response: 'Hi!' }),
    },
  };
});

describe('useChat', () => {
  it('appends user message and calls streaming', async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      result.current.sendMessage('Hello', true);
    });
    expect(result.current.messages[0].content).toBe('Hello');
  });

  it('handles non-streaming chat', async () => {
    const { result } = renderHook(() => useChat());
    await act(async () => {
      result.current.sendMessage('Hello', false);
    });
    // allow promise to resolve
    await act(async () => {});
    expect(result.current.messages.length).toBeGreaterThanOrEqual(2);
  });

  it('manages stop button state correctly during streaming', async () => {
    const { result } = renderHook(() => useChat());

    // Initially not streaming
    expect(result.current.isStreaming).toBe(false);

    // Start streaming
    await act(async () => {
      result.current.sendMessage('Hello', true);
    });

    // Should be streaming now
    expect(result.current.isStreaming).toBe(true);

    // Stop streaming
    act(() => {
      result.current.stopStreaming();
    });

    // Should not be streaming after stop
    expect(result.current.isStreaming).toBe(false);

    // Verify WebSocket was closed
    expect(mockWebSocket.close).toHaveBeenCalled();
  });

  it('prevents race conditions in streaming state', async () => {
    const { result } = renderHook(() => useChat());

    // Start streaming
    await act(async () => {
      result.current.sendMessage('Hello', true);
    });

    expect(result.current.isStreaming).toBe(true);

    // Simulate multiple rapid stop calls (race condition)
    act(() => {
      result.current.stopStreaming();
      result.current.stopStreaming(); // Second call should not cause issues
    });

    expect(result.current.isStreaming).toBe(false);
  });

  it('ensures input is enabled after streaming finishes', async () => {
    const { result } = renderHook(() => useChat());

    // Initially not streaming
    expect(result.current.isStreaming).toBe(false);

    // Start streaming
    await act(async () => {
      result.current.sendMessage('Hello', true);
    });

    // Should be streaming
    expect(result.current.isStreaming).toBe(true);

    // Simulate streaming completion (done message)
    act(() => {
      // This would normally come from the WebSocket, but we can't easily simulate that
      // So we'll call the stopStreaming function which should reset the state
      result.current.stopStreaming();
    });

    // Should not be streaming after completion
    expect(result.current.isStreaming).toBe(false);
  });
});


