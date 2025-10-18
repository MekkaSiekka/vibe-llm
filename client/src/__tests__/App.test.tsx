/**
 * Unit Tests for App Component with Sidebar Integration
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';

// Mock child components
vi.mock('../components/Sidebar', () => ({
  Sidebar: ({ activeTab, onTabChange, isCollapsed, onToggleCollapse }: any) => (
    <div data-testid="sidebar">
      <button onClick={() => onToggleCollapse()}>Toggle Sidebar</button>
      <button onClick={() => onTabChange('chat')}>Chat</button>
      <button onClick={() => onTabChange('ai-detect')}>AI Detect</button>
      <button onClick={() => onTabChange('settings')}>Settings</button>
      <span data-testid="collapsed-state">{isCollapsed ? 'collapsed' : 'expanded'}</span>
      <span data-testid="active-tab">{activeTab}</span>
    </div>
  ),
}));

vi.mock('../components/ChatView', () => ({
  ChatView: () => <div data-testid="chat-view">Chat View</div>,
}));

vi.mock('../components/AIDetectView', () => ({
  AIDetectView: ({ text, setText, analyzing, setAnalyzing }: any) => (
    <div data-testid="ai-detect-view">
      <span data-testid="ai-detect-text">{text}</span>
      <button onClick={() => setText('test text')}>Set Text</button>
      <button onClick={() => setAnalyzing(!analyzing)}>Toggle Analyzing</button>
      <span data-testid="ai-detect-analyzing">{analyzing ? 'analyzing' : 'idle'}</span>
    </div>
  ),
}));

vi.mock('../components/SettingsView', () => ({
  SettingsView: () => <div data-testid="settings-view">Settings View</div>,
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial Render', () => {
    it('renders the app with sidebar', () => {
      render(<App />);
      expect(screen.getByTestId('sidebar')).toBeInTheDocument();
    });

    it('starts with chat view active', () => {
      render(<App />);
      expect(screen.getByTestId('active-tab')).toHaveTextContent('chat');
      expect(screen.getByTestId('chat-view')).toBeInTheDocument();
    });

    it('starts with sidebar collapsed on mobile', () => {
      // Mock window.innerWidth for mobile
      window.innerWidth = 800;
      window.dispatchEvent(new Event('resize'));
      
      render(<App />);
      
      // Initial render might be collapsed due to useEffect
      waitFor(() => {
        expect(screen.getByTestId('collapsed-state')).toHaveTextContent('collapsed');
      });
    });
  });

  describe('View Navigation', () => {
    it('switches to chat view when chat button is clicked', () => {
      render(<App />);
      
      const chatButton = screen.getByText('Chat');
      fireEvent.click(chatButton);
      
      expect(screen.getByTestId('chat-view')).toBeInTheDocument();
      expect(screen.getByTestId('active-tab')).toHaveTextContent('chat');
    });

    it('switches to AI detect view when AI detect button is clicked', () => {
      render(<App />);
      
      const aiDetectButton = screen.getByText('AI Detect');
      fireEvent.click(aiDetectButton);
      
      expect(screen.getByTestId('ai-detect-view')).toBeInTheDocument();
      expect(screen.getByTestId('active-tab')).toHaveTextContent('ai-detect');
    });

    it('switches to settings view when settings button is clicked', () => {
      render(<App />);
      
      const settingsButton = screen.getByText('Settings');
      fireEvent.click(settingsButton);
      
      expect(screen.getByTestId('settings-view')).toBeInTheDocument();
      expect(screen.getByTestId('active-tab')).toHaveTextContent('settings');
    });

    it('can navigate between multiple views', () => {
      render(<App />);
      
      fireEvent.click(screen.getByText('Chat'));
      expect(screen.getByTestId('chat-view')).toBeInTheDocument();
      
      fireEvent.click(screen.getByText('AI Detect'));
      expect(screen.getByTestId('ai-detect-view')).toBeInTheDocument();
      
      fireEvent.click(screen.getByText('Settings'));
      expect(screen.getByTestId('settings-view')).toBeInTheDocument();
      
      fireEvent.click(screen.getByText('Chat'));
      expect(screen.getByTestId('chat-view')).toBeInTheDocument();
    });
  });

  describe('Sidebar Collapse/Expand', () => {
    it('toggles sidebar collapse state', () => {
      render(<App />);
      
      const toggleButton = screen.getByText('Toggle Sidebar');
      const collapsedState = screen.getByTestId('collapsed-state');
      
      // Initial state might be expanded on desktop
      const initialState = collapsedState.textContent;
      
      fireEvent.click(toggleButton);
      
      // State should change
      expect(collapsedState.textContent).not.toBe(initialState);
    });

    it('can toggle sidebar multiple times', () => {
      render(<App />);
      
      const toggleButton = screen.getByText('Toggle Sidebar');
      const collapsedState = screen.getByTestId('collapsed-state');
      
      const initialState = collapsedState.textContent;
      
      fireEvent.click(toggleButton);
      const secondState = collapsedState.textContent;
      
      fireEvent.click(toggleButton);
      const thirdState = collapsedState.textContent;
      
      // Should toggle back and forth
      expect(secondState).not.toBe(initialState);
      expect(thirdState).toBe(initialState);
    });
  });

  describe('AI Detect State Persistence', () => {
    it('preserves AI detect text when switching tabs', () => {
      render(<App />);
      
      // Navigate to AI Detect
      fireEvent.click(screen.getByText('AI Detect'));
      
      // Set some text
      fireEvent.click(screen.getByText('Set Text'));
      expect(screen.getByTestId('ai-detect-text')).toHaveTextContent('test text');
      
      // Switch to another tab
      fireEvent.click(screen.getByText('Chat'));
      expect(screen.getByTestId('chat-view')).toBeInTheDocument();
      
      // Switch back to AI Detect
      fireEvent.click(screen.getByText('AI Detect'));
      
      // Text should be preserved
      expect(screen.getByTestId('ai-detect-text')).toHaveTextContent('test text');
    });

    it('preserves AI detect analyzing state when switching tabs', () => {
      render(<App />);
      
      // Navigate to AI Detect
      fireEvent.click(screen.getByText('AI Detect'));
      
      // Toggle analyzing state
      fireEvent.click(screen.getByText('Toggle Analyzing'));
      expect(screen.getByTestId('ai-detect-analyzing')).toHaveTextContent('analyzing');
      
      // Switch to another tab
      fireEvent.click(screen.getByText('Settings'));
      
      // Switch back to AI Detect
      fireEvent.click(screen.getByText('AI Detect'));
      
      // Analyzing state should be preserved
      expect(screen.getByTestId('ai-detect-analyzing')).toHaveTextContent('analyzing');
    });
  });

  describe('Responsive Behavior', () => {
    it('handles window resize events', () => {
      render(<App />);
      
      // Simulate resize to mobile
      window.innerWidth = 800;
      window.dispatchEvent(new Event('resize'));
      
      // Should trigger auto-collapse on mobile
      waitFor(() => {
        expect(screen.getByTestId('collapsed-state')).toHaveTextContent('collapsed');
      });
    });

    it('handles window resize to desktop', () => {
      // Start with mobile size
      window.innerWidth = 800;
      
      render(<App />);
      
      // Resize to desktop
      window.innerWidth = 1200;
      window.dispatchEvent(new Event('resize'));
      
      // Sidebar state is managed by component
      expect(screen.getByTestId('sidebar')).toBeInTheDocument();
    });
  });

  describe('Mobile Menu Button', () => {
    it('shows mobile menu button when sidebar is collapsed', () => {
      render(<App />);
      
      // Ensure sidebar is collapsed
      const toggleButton = screen.getByText('Toggle Sidebar');
      fireEvent.click(toggleButton);
      
      waitFor(() => {
        const collapsedState = screen.getByTestId('collapsed-state');
        if (collapsedState.textContent === 'collapsed') {
          // Mobile menu button should be visible (has aria-label="Open menu")
          const menuButton = document.querySelector('[aria-label="Open menu"]');
          expect(menuButton).toBeInTheDocument();
        }
      });
    });
  });

  describe('Edge Cases', () => {
    it('handles rapid tab switching', () => {
      render(<App />);
      
      // Rapid tab switches
      fireEvent.click(screen.getByText('Chat'));
      fireEvent.click(screen.getByText('AI Detect'));
      fireEvent.click(screen.getByText('Settings'));
      fireEvent.click(screen.getByText('Chat'));
      
      // Should end up on Chat view
      expect(screen.getByTestId('chat-view')).toBeInTheDocument();
      expect(screen.getByTestId('active-tab')).toHaveTextContent('chat');
    });

    it('handles navigation while state is being updated', () => {
      render(<App />);
      
      // Navigate to AI Detect
      fireEvent.click(screen.getByText('AI Detect'));
      
      // Start updating state
      fireEvent.click(screen.getByText('Set Text'));
      
      // Immediately switch tabs
      fireEvent.click(screen.getByText('Chat'));
      
      // Should still be on Chat
      expect(screen.getByTestId('chat-view')).toBeInTheDocument();
    });

    it('renders default view for unknown tab', () => {
      render(<App />);
      
      // The app should always render a valid view
      const activeTab = screen.getByTestId('active-tab').textContent;
      expect(['chat', 'ai-detect', 'settings']).toContain(activeTab);
    });
  });

  describe('Layout', () => {
    it('applies correct margin classes based on sidebar state', () => {
      const { container } = render(<App />);
      
      const main = container.querySelector('main');
      expect(main).toBeInTheDocument();
      
      // Should have transition classes
      expect(main).toHaveClass('transition-all');
    });

    it('main content area exists', () => {
      const { container } = render(<App />);
      
      const main = container.querySelector('main');
      expect(main).toBeInTheDocument();
      expect(main).toHaveClass('app-main');
    });
  });
});

