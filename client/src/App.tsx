/**
 * Main App Component with Sidebar Navigation
 */

import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatView } from './components/ChatView';
import { AIDetectView } from './components/AIDetectView';
import { SettingsView } from './components/SettingsView';
import { cn } from './lib/utils';
import type { Tab } from './types';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  // AI Detect state - lifted to preserve across tab switches
  const [aiDetectText, setAiDetectText] = useState('');
  const [aiDetectAnalyzing, setAiDetectAnalyzing] = useState(false);

  // Auto-collapse sidebar on mobile
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true);
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Close sidebar on mobile when tab changes
  const handleTabChange = (tab: Tab) => {
    setActiveTab(tab);
    if (window.innerWidth < 1024 && !sidebarCollapsed) {
      setSidebarCollapsed(true);
    }
  };

  const renderView = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatView />;
      case 'ai-detect':
        return (
          <AIDetectView
            text={aiDetectText}
            setText={setAiDetectText}
            analyzing={aiDetectAnalyzing}
            setAnalyzing={setAiDetectAnalyzing}
          />
        );
      case 'settings':
        return <SettingsView />;
      default:
        return <ChatView />;
    }
  };

  return (
    <div className="app">
      <Sidebar 
        activeTab={activeTab} 
        onTabChange={handleTabChange}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      
      <main 
        className={cn(
          "app-main transition-all duration-300",
          sidebarCollapsed ? "ml-16" : "ml-64"
        )}
      >
        {renderView()}
      </main>

      {/* Mobile menu button */}
      {sidebarCollapsed && (
        <button
          onClick={() => setSidebarCollapsed(false)}
          className={cn(
            "lg:hidden fixed top-4 left-4 z-30",
            "w-10 h-10 flex items-center justify-center",
            "bg-card border border-border rounded-ios-md shadow-ios-lg",
            "text-foreground hover:bg-muted transition-all duration-200",
            "active:scale-95"
          )}
          aria-label="Open menu"
        >
          <span className="text-xl">☰</span>
        </button>
      )}
    </div>
  );
}

export default App;

