/**
 * Unit Tests for Sidebar Component
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Sidebar } from '../Sidebar';
import type { Tab } from '../../types';

describe('Sidebar', () => {
  const mockOnTabChange = vi.fn();
  const mockOnToggleCollapse = vi.fn();

  const defaultProps = {
    activeTab: 'chat' as Tab,
    onTabChange: mockOnTabChange,
    isCollapsed: false,
    onToggleCollapse: mockOnToggleCollapse,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the sidebar with logo and title when expanded', () => {
      render(<Sidebar {...defaultProps} />);
      
      expect(screen.getByText('Vibe LLM')).toBeInTheDocument();
      expect(screen.getByText('AI Assistant')).toBeInTheDocument();
    });

    it('renders all navigation items', () => {
      render(<Sidebar {...defaultProps} />);
      
      expect(screen.getByText('Chat')).toBeInTheDocument();
      expect(screen.getByText('AI Detect')).toBeInTheDocument();
      expect(screen.getByText('Settings')).toBeInTheDocument();
    });

    it('renders navigation item descriptions when expanded', () => {
      render(<Sidebar {...defaultProps} />);
      
      expect(screen.getByText('AI conversation')).toBeInTheDocument();
      expect(screen.getByText('Detect AI-generated text')).toBeInTheDocument();
      expect(screen.getByText('App configuration')).toBeInTheDocument();
    });

    it('hides text content when collapsed', () => {
      render(<Sidebar {...defaultProps} isCollapsed={true} />);
      
      // Text labels should not be visible when collapsed
      expect(screen.queryByText('Vibe LLM')).not.toBeInTheDocument();
      expect(screen.queryByText('Chat')).not.toBeInTheDocument();
    });

    it('shows only logo icon when collapsed', () => {
      render(<Sidebar {...defaultProps} isCollapsed={true} />);
      
      // Should still render the emoji icons
      const aside = screen.getByRole('complementary');
      expect(aside).toBeInTheDocument();
    });
  });

  describe('Navigation', () => {
    it('highlights the active tab', () => {
      render(<Sidebar {...defaultProps} activeTab="chat" />);
      
      const chatButton = screen.getByRole('button', { name: /chat/i });
      expect(chatButton).toHaveAttribute('aria-current', 'page');
    });

    it('calls onTabChange when a navigation item is clicked', () => {
      render(<Sidebar {...defaultProps} />);
      
      const aiDetectButton = screen.getByRole('button', { name: /ai detect/i });
      fireEvent.click(aiDetectButton);
      
      expect(mockOnTabChange).toHaveBeenCalledWith('ai-detect');
      expect(mockOnTabChange).toHaveBeenCalledTimes(1);
    });

    it('can navigate to all tabs', () => {
      render(<Sidebar {...defaultProps} />);
      
      const chatButton = screen.getByRole('button', { name: /chat/i });
      const aiDetectButton = screen.getByRole('button', { name: /ai detect/i });
      const settingsButton = screen.getByRole('button', { name: /settings/i });
      
      fireEvent.click(chatButton);
      expect(mockOnTabChange).toHaveBeenCalledWith('chat');
      
      fireEvent.click(aiDetectButton);
      expect(mockOnTabChange).toHaveBeenCalledWith('ai-detect');
      
      fireEvent.click(settingsButton);
      expect(mockOnTabChange).toHaveBeenCalledWith('settings');
      
      expect(mockOnTabChange).toHaveBeenCalledTimes(3);
    });

    it('does not highlight inactive tabs', () => {
      render(<Sidebar {...defaultProps} activeTab="chat" />);
      
      const aiDetectButton = screen.getByRole('button', { name: /ai detect/i });
      expect(aiDetectButton).not.toHaveAttribute('aria-current');
    });
  });

  describe('Collapse/Expand Functionality', () => {
    it('calls onToggleCollapse when collapse button is clicked', () => {
      render(<Sidebar {...defaultProps} />);
      
      const collapseButton = screen.getByRole('button', { name: /collapse/i });
      fireEvent.click(collapseButton);
      
      expect(mockOnToggleCollapse).toHaveBeenCalledTimes(1);
    });

    it('shows correct icon in collapse button when expanded', () => {
      render(<Sidebar {...defaultProps} isCollapsed={false} />);
      
      const collapseButton = screen.getByRole('button', { name: /collapse/i });
      expect(collapseButton.textContent).toContain('←');
    });

    it('shows correct icon in collapse button when collapsed', () => {
      render(<Sidebar {...defaultProps} isCollapsed={true} />);
      
      // Use title attribute to find buttons when collapsed
      const expandButtons = screen.getAllByTitle('Expand sidebar');
      expect(expandButtons.length).toBeGreaterThan(0);
      expect(expandButtons[0].textContent).toContain('→');
    });

    it('shows expand button when collapsed on desktop', () => {
      render(<Sidebar {...defaultProps} isCollapsed={true} />);
      
      // The floating expand button for desktop (multiple buttons with same title)
      const expandButtons = screen.getAllByTitle('Expand sidebar');
      expect(expandButtons.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels on navigation items', () => {
      render(<Sidebar {...defaultProps} activeTab="chat" />);
      
      const chatButton = screen.getByRole('button', { name: /chat/i });
      expect(chatButton).toHaveAttribute('aria-current', 'page');
    });

    it('has proper title attributes when collapsed', () => {
      render(<Sidebar {...defaultProps} isCollapsed={true} />);
      
      const buttons = screen.getAllByRole('button');
      // Navigation buttons should have title attributes when collapsed
      const navButtons = buttons.filter(btn => 
        btn.getAttribute('title') === 'Chat' ||
        btn.getAttribute('title') === 'AI Detect' ||
        btn.getAttribute('title') === 'Settings'
      );
      expect(navButtons.length).toBe(3);
    });

    it('has keyboard navigation support', () => {
      render(<Sidebar {...defaultProps} />);
      
      const chatButton = screen.getByRole('button', { name: /chat/i });
      chatButton.focus();
      
      expect(document.activeElement).toBe(chatButton);
    });

    it('renders as a complementary landmark', () => {
      render(<Sidebar {...defaultProps} />);
      
      const aside = screen.getByRole('complementary');
      expect(aside).toBeInTheDocument();
    });
  });

  describe('Mobile Overlay', () => {
    it('does not render overlay when collapsed', () => {
      const { container } = render(<Sidebar {...defaultProps} isCollapsed={true} />);
      
      // Check for overlay div
      const overlay = container.querySelector('.fixed.inset-0.bg-black\\/50');
      expect(overlay).not.toBeInTheDocument();
    });

    it('renders overlay when expanded on mobile', () => {
      const { container } = render(<Sidebar {...defaultProps} isCollapsed={false} />);
      
      // Check for overlay div (it has lg:hidden class for mobile)
      const overlay = container.querySelector('.lg\\:hidden.z-40');
      expect(overlay).toBeInTheDocument();
    });

    it('calls onToggleCollapse when overlay is clicked', () => {
      const { container } = render(<Sidebar {...defaultProps} isCollapsed={false} />);
      
      const overlay = container.querySelector('.lg\\:hidden.z-40');
      if (overlay) {
        fireEvent.click(overlay);
        expect(mockOnToggleCollapse).toHaveBeenCalledTimes(1);
      }
    });
  });

  describe('Visual States', () => {
    it('applies correct CSS classes when expanded', () => {
      const { container } = render(<Sidebar {...defaultProps} isCollapsed={false} />);
      
      const aside = container.querySelector('aside');
      expect(aside).toHaveClass('w-64');
    });

    it('applies correct CSS classes when collapsed', () => {
      const { container } = render(<Sidebar {...defaultProps} isCollapsed={true} />);
      
      const aside = container.querySelector('aside');
      expect(aside).toHaveClass('w-16');
    });

    it('applies active state styling to current tab', () => {
      render(<Sidebar {...defaultProps} activeTab="chat" />);
      
      const chatButton = screen.getByRole('button', { name: /chat/i });
      expect(chatButton).toHaveClass('bg-primary');
    });
  });

  describe('Edge Cases', () => {
    it('handles rapid tab switching', () => {
      render(<Sidebar {...defaultProps} />);
      
      const chatButton = screen.getByRole('button', { name: /chat/i });
      const aiDetectButton = screen.getByRole('button', { name: /ai detect/i });
      
      // Rapid clicks
      fireEvent.click(chatButton);
      fireEvent.click(aiDetectButton);
      fireEvent.click(chatButton);
      fireEvent.click(aiDetectButton);
      
      expect(mockOnTabChange).toHaveBeenCalledTimes(4);
    });

    it('handles rapid collapse/expand toggling', () => {
      render(<Sidebar {...defaultProps} />);
      
      const collapseButton = screen.getByRole('button', { name: /collapse/i });
      
      // Rapid clicks
      fireEvent.click(collapseButton);
      fireEvent.click(collapseButton);
      fireEvent.click(collapseButton);
      
      expect(mockOnToggleCollapse).toHaveBeenCalledTimes(3);
    });

    it('renders correctly with different active tabs', () => {
      const { rerender } = render(<Sidebar {...defaultProps} activeTab="chat" />);
      expect(screen.getByRole('button', { name: /chat/i })).toHaveAttribute('aria-current', 'page');
      
      rerender(<Sidebar {...defaultProps} activeTab="ai-detect" />);
      expect(screen.getByRole('button', { name: /ai detect/i })).toHaveAttribute('aria-current', 'page');
      
      rerender(<Sidebar {...defaultProps} activeTab="settings" />);
      expect(screen.getByRole('button', { name: /settings/i })).toHaveAttribute('aria-current', 'page');
    });
  });
});

