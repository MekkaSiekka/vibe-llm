import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { AIDetectView } from '../AIDetectView';

describe('AIDetectView', () => {
  it('renders with initial props', () => {
    const setText = vi.fn();
    const setAnalyzing = vi.fn();
    
    render(
      <AIDetectView
        text=""
        setText={setText}
        analyzing={false}
        setAnalyzing={setAnalyzing}
      />
    );
    
    expect(screen.getByText('AI Text Detection')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Paste text here to analyze...')).toBeInTheDocument();
  });

  it('calls setText when input changes', () => {
    const setText = vi.fn();
    const setAnalyzing = vi.fn();
    
    render(
      <AIDetectView
        text=""
        setText={setText}
        analyzing={false}
        setAnalyzing={setAnalyzing}
      />
    );
    
    const textarea = screen.getByPlaceholderText('Paste text here to analyze...') as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'Test text' } });
    
    expect(setText).toHaveBeenCalledWith('Test text');
  });

  it('displays provided text value', () => {
    const setText = vi.fn();
    const setAnalyzing = vi.fn();
    
    render(
      <AIDetectView
        text="Existing text"
        setText={setText}
        analyzing={false}
        setAnalyzing={setAnalyzing}
      />
    );
    
    const textarea = screen.getByPlaceholderText('Paste text here to analyze...') as HTMLTextAreaElement;
    expect(textarea.value).toBe('Existing text');
  });

  it('disables analyze button when text is empty', async () => {
    const setText = vi.fn();
    const setAnalyzing = vi.fn();
    
    const { rerender } = render(
      <AIDetectView
        text="Some text"
        setText={setText}
        analyzing={false}
        setAnalyzing={setAnalyzing}
      />
    );
    
    // Initially with text, button should be enabled
    const analyzeButton = screen.getByRole('button', { name: /analyze/i });
    expect(analyzeButton).not.toBeDisabled();
    
    // When text is empty, analyze should be disabled
    rerender(
      <AIDetectView
        text=""
        setText={setText}
        analyzing={false}
        setAnalyzing={setAnalyzing}
      />
    );
    
    const analyzeButtonEmpty = screen.getByRole('button', { name: /analyze/i });
    expect(analyzeButtonEmpty).toBeDisabled();
  });
});

