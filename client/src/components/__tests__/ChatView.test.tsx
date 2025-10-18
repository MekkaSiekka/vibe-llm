import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ChatView } from '../ChatView';

const mockSendMessage = vi.fn();
const mockStopStreaming = vi.fn();
const mockClearMessages = vi.fn();

// Mock useChat hook
vi.mock('../../hooks/useChat', () => ({
  useChat: () => ({
    messages: [],
    isStreaming: false,
    error: null,
    sendMessage: mockSendMessage,
    stopStreaming: mockStopStreaming,
    clearMessages: mockClearMessages,
  }),
}));

describe('ChatView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders header and input', () => {
    render(<ChatView />);
    expect(screen.getByText('Chat')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Message...')).toBeInTheDocument();
  });

  it('sends message on enter', () => {
    render(<ChatView />);
    const input = screen.getByPlaceholderText('Message...') as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    expect(mockSendMessage).toHaveBeenCalledWith('Hello');
  });
});


