/**
 * Modern Chat Message Bubble Component (2025 iOS Design)
 */

import { ChatMessage } from '../types';
import { formatTime } from '../lib/utils';
import { cn } from '../lib/utils';

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  
  return (
    <div
      className={cn(
        "flex gap-2 mb-4 animate-slide-up",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-xs lg:max-w-md px-4 py-2 rounded-ios-lg text-ios-base leading-relaxed break-words",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-sm"
            : "bg-card border border-border text-foreground rounded-bl-sm"
        )}
      >
        {message.content}
      </div>
      <div className="text-ios-xs text-muted-foreground self-end mb-1 px-1">
        {formatTime(message.timestamp)}
      </div>
    </div>
  );
}

