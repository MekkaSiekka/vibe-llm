/**
 * Modern Collapsible Sidebar Navigation (2025 Design)
 */

import { cn } from '../lib/utils';
import type { Tab } from '../types';

interface SidebarProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

interface NavItem {
  id: Tab;
  label: string;
  icon: string;
  description: string;
}

export function Sidebar({ activeTab, onTabChange, isCollapsed, onToggleCollapse }: SidebarProps) {
  const navItems: NavItem[] = [
    { 
      id: 'chat', 
      label: 'Chat', 
      icon: '💬',
      description: 'AI conversation'
    },
    { 
      id: 'ai-detect', 
      label: 'AI Detect', 
      icon: '🔍',
      description: 'Detect AI-generated text'
    },
    { 
      id: 'settings', 
      label: 'Settings', 
      icon: '⚙️',
      description: 'App configuration'
    },
  ];

  return (
    <>
      {/* Overlay for mobile */}
      {!isCollapsed && (
        <div 
          className="fixed inset-0 bg-black/50 lg:hidden z-40 animate-fade-in"
          onClick={onToggleCollapse}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed top-0 left-0 h-full bg-card border-r border-border z-50",
          "flex flex-col transition-all duration-300 ease-in-out",
          "shadow-ios-xl",
          isCollapsed ? "w-16" : "w-64"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-border">
          {!isCollapsed && (
            <div className="flex items-center gap-3 animate-slide-right">
              <div className="w-8 h-8 bg-primary rounded-ios-md flex items-center justify-center text-lg">
                🚀
              </div>
              <div>
                <h1 className="text-ios-lg font-bold text-foreground">Vibe LLM</h1>
                <p className="text-ios-xs text-muted-foreground">AI Assistant</p>
              </div>
            </div>
          )}
          
          {isCollapsed && (
            <div className="w-full flex justify-center animate-fade-in">
              <div className="w-8 h-8 bg-primary rounded-ios-md flex items-center justify-center text-lg">
                🚀
              </div>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4 px-2">
          <div className="space-y-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-3 rounded-ios-lg",
                  "transition-all duration-200 active:scale-95",
                  "text-ios-base font-medium",
                  activeTab === item.id
                    ? "bg-primary text-primary-foreground shadow-ios-sm"
                    : "text-foreground hover:bg-muted"
                )}
                aria-current={activeTab === item.id ? 'page' : undefined}
                title={isCollapsed ? item.label : undefined}
              >
                <span className="text-2xl leading-none flex-shrink-0">{item.icon}</span>
                {!isCollapsed && (
                  <div className="flex-1 text-left animate-slide-right overflow-hidden">
                    <div className="truncate">{item.label}</div>
                    <div className={cn(
                      "text-ios-xs truncate",
                      activeTab === item.id 
                        ? "text-primary-foreground/80" 
                        : "text-muted-foreground"
                    )}>
                      {item.description}
                    </div>
                  </div>
                )}
              </button>
            ))}
          </div>
        </nav>

        {/* Footer - Toggle Button */}
        <div className="border-t border-border p-2">
          <button
            onClick={onToggleCollapse}
            className={cn(
              "w-full flex items-center gap-3 px-3 py-3 rounded-ios-lg",
              "transition-all duration-200 active:scale-95",
              "text-foreground hover:bg-muted text-ios-sm font-medium"
            )}
            title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <span className="text-xl leading-none flex-shrink-0">
              {isCollapsed ? '→' : '←'}
            </span>
            {!isCollapsed && (
              <span className="animate-slide-right">Collapse</span>
            )}
          </button>
        </div>
      </aside>

      {/* Toggle button for collapsed state (desktop) */}
      {isCollapsed && (
        <button
          onClick={onToggleCollapse}
          className={cn(
            "hidden lg:flex fixed top-4 left-20 z-30",
            "w-8 h-8 items-center justify-center",
            "bg-card border border-border rounded-ios-md shadow-ios-md",
            "text-foreground hover:bg-muted transition-all duration-200",
            "active:scale-95"
          )}
          title="Expand sidebar"
        >
          <span className="text-lg">→</span>
        </button>
      )}
    </>
  );
}

