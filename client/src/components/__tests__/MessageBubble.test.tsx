import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MessageBubble } from '../MessageBubble';

describe('MessageBubble', () => {
  it('renders user message', () => {
    render(
      <MessageBubble message={{ role: 'user', content: 'Hello', timestamp: new Date() }} />
    );
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('renders assistant message', () => {
    render(
      <MessageBubble message={{ role: 'assistant', content: 'Hi there', timestamp: new Date() }} />
    );
    expect(screen.getByText('Hi there')).toBeInTheDocument();
  });
});


