/**
 * Modern Chat View Component with Streaming (2025 iOS Design)
 */

import { useState, useRef, useEffect, FormEvent } from 'react';
import { useChat } from '../hooks/useChat';
import { MessageBubble } from './MessageBubble';
import { Button } from './Button';
import { ModelSelector } from './ModelSelector';
import { cn } from '../lib/utils';

export function ChatView() {
  const { messages, isStreaming, error, sendMessage, stopStreaming, clearMessages } = useChat();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);


  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    sendMessage(input);
    setInput('');
    if (inputRef.current) inputRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="flex justify-between items-center px-6 py-4 bg-card border-b border-border">
        <div className="flex-1">
          <h2 className="text-ios-2xl font-bold text-foreground">Chat</h2>
          <p className="text-ios-sm text-muted-foreground mt-1">Powered by Vibe LLM</p>
        </div>
        <div className="flex items-center gap-2">
          <ModelSelector compact />
          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearMessages}
              disabled={isStreaming}
            >
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 flex flex-col">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center text-center py-16 my-auto">
            <div className="text-5xl mb-4 opacity-60">💬</div>
            <h3 className="text-ios-2xl font-semibold text-foreground mb-2">Start a conversation</h3>
            <p className="text-ios-base text-muted-foreground max-w-xs">
              Ask me anything! I'm here to help with your questions.
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))}

        {error && (
          <div className="flex items-start gap-3 p-4 bg-destructive/10 border border-destructive rounded-ios-lg mb-4 animate-slide-up">
            <span className="text-lg">⚠️</span>
            <span className="text-ios-sm text-destructive">{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-card border-t border-border p-4">
        <form onSubmit={handleSubmit} className={cn(
          "flex gap-2 items-end bg-secondary rounded-ios-lg border border-border p-2",
          "focus-within:border-primary transition-colors"
        )}>
          <textarea
            ref={inputRef}
            className="flex-1 min-h-10 max-h-32 px-3 py-2 bg-transparent text-ios-base text-foreground resize-none outline-none font-sans"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message..."
            rows={1}
            disabled={isStreaming}
          />
          <div className="flex items-center">
            {isStreaming ? (
              <Button
                variant="secondary"
                size="sm"
                onClick={stopStreaming}
                type="button"
              >
                Stop
              </Button>
            ) : (
              <Button
                variant="primary"
                size="sm"
                type="submit"
                disabled={!input.trim()}
                icon="📤"
              >
                Send
              </Button>
            )}
          </div>
        </form>
        <p className="text-ios-xs text-muted-foreground text-center mt-2">
          Press <kbd className="px-1.5 py-0.5 bg-secondary border border-border rounded text-foreground font-mono text-ios-xs">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 bg-secondary border border-border rounded text-foreground font-mono text-ios-xs">Shift + Enter</kbd> for new line
        </p>
      </div>
    </div>
  );
}

