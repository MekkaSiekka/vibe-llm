/**
 * Custom hook for chat functionality with streaming support
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { api } from '../services/api';
import type { ChatMessage, WebSocketMessage } from '../types';

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamingFinishedRef = useRef(false);

  // Safety effect to ensure isStreaming is properly reset
  useEffect(() => {
    if (isStreaming && !streamingFinishedRef.current) {
      // If we're streaming but the finished ref is false, something might be wrong
      // This is a safety check to prevent hanging states
      const safetyCheck = setTimeout(() => {
        if (isStreaming && !streamingFinishedRef.current) {
          console.warn('Streaming state safety check triggered, resetting state');
          streamingFinishedRef.current = true;
          setIsStreaming(false);
        }
      }, 10000); // 10 second safety timeout

      return () => clearTimeout(safetyCheck);
    }
  }, [isStreaming]);

  const sendMessage = useCallback((content: string, useStreaming = true) => {
    if (!content.trim()) return;

    // Prepare conversation history BEFORE adding new messages
    // Use current messages state (all previous conversation)
    const conversationHistory = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    console.log('Sending message with conversation history:', conversationHistory.length, 'messages');

    // Add user message
    const userMessage: ChatMessage = {
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setError(null);

    if (useStreaming) {
      // Reset streaming state for new message
      streamingFinishedRef.current = false;
      setIsStreaming(true);
      setError(null);
      let accumulatedContent = '';

      // Add placeholder for assistant message
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Create a safety timeout that forces streaming to stop
      const forceStopTimeout = setTimeout(() => {
        streamingFinishedRef.current = true;
        setIsStreaming(false);
      }, 15000); // 15 second timeout

      wsRef.current = api.streamChat(
        content,
        (msg: WebSocketMessage) => {
          if (streamingFinishedRef.current) {
            return; // Ignore messages if streaming already finished
          }

          if (msg.type === 'chunk') {
            accumulatedContent += msg.content;
            setMessages(prev => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = {
                ...newMessages[newMessages.length - 1],
                content: accumulatedContent,
              };
              return newMessages;
            });
          } else if (msg.type === 'done' || msg.type === 'complete') {
            clearTimeout(forceStopTimeout);
            streamingFinishedRef.current = true;
            setIsStreaming(false);
          } else if (msg.type === 'error') {
            clearTimeout(forceStopTimeout);
            streamingFinishedRef.current = true;
            setError(msg.content);
            setIsStreaming(false);
          }
        },
        (err) => {
          clearTimeout(forceStopTimeout);
          streamingFinishedRef.current = true;
          setError(err.message);
          setIsStreaming(false);
        },
        () => {
          clearTimeout(forceStopTimeout);
          streamingFinishedRef.current = true;
          setIsStreaming(false);
        },
        {
          conversationHistory: conversationHistory
        }
      );
    } else {
      // Simple non-streaming chat
      api.chat({ message: content })
        .then(response => {
          if (response.success) {
            const assistantMessage: ChatMessage = {
              role: 'assistant',
              content: response.response,
              timestamp: new Date(),
            };
            setMessages(prev => [...prev, assistantMessage]);
          } else {
            setError(response.error || 'Chat failed');
          }
        })
        .catch(err => {
          setError(err.message);
        });
    }
  }, [messages]);

  const stopStreaming = useCallback(() => {
    if (wsRef.current) {
      streamingFinishedRef.current = true;
      wsRef.current.close();
      wsRef.current = null;
    }
    // Force reset the streaming state
    setIsStreaming(false);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
    clearMessages,
  };
}

