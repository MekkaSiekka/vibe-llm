/**
 * Model Selector Component - Dropdown to select and load models
 */

import { useState, useRef, useEffect } from 'react';
import { useModels } from '../hooks/useModels';
import type { ModelInfo } from '../types';
import './ModelSelector.css';

interface ModelSelectorProps {
  onModelChange?: (modelName: string) => void;
  compact?: boolean;
  modelType?: 'chat' | 'ai_detector'; // Filter by model type
}

export function ModelSelector({ onModelChange, compact = false, modelType = 'chat' }: ModelSelectorProps) {
  const { models, currentModel, loading, loadModel } = useModels();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const filteredModels = models.filter(m => m.model_type === modelType);
  
  // Find currently loaded model of this type (either by name match or by loaded flag)
  const currentModelInfo = filteredModels.find(m => m.name === currentModel) 
                        || filteredModels.find(m => m.loaded);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelectModel = async (modelName: string) => {
    setIsOpen(false);
    const result = await loadModel(modelName);
    if (result.success && onModelChange) {
      onModelChange(modelName);
    }
  };

  return (
    <div className={`model-selector ${compact ? 'model-selector-compact' : ''}`} ref={dropdownRef}>
      <button
        className="model-selector-trigger"
        onClick={() => setIsOpen(!isOpen)}
        disabled={loading}
      >
        <span className="model-selector-icon">🤖</span>
        <span className="model-selector-label">
          {loading ? 'Loading...' : currentModelInfo?.name || 'Select Model'}
        </span>
        <span className={`model-selector-arrow ${isOpen ? 'model-selector-arrow-up' : ''}`}>
          ▼
        </span>
      </button>

      {isOpen && (
        <div className="model-selector-dropdown">
          <div className="model-selector-header">
            <h3>{modelType === 'chat' ? 'Chat Models' : 'AI Detection Models'}</h3>
            {currentModel && (
              <span className="model-selector-current">Current</span>
            )}
          </div>
          
          <div className="model-selector-list">
            {filteredModels.length === 0 && (
              <div className="model-selector-empty">
                <p>No {modelType === 'chat' ? 'chat' : 'detection'} models available</p>
              </div>
            )}
            
            {filteredModels.map((model) => (
              <ModelOption
                key={model.name}
                model={model}
                isActive={model.name === currentModel}
                onSelect={handleSelectModel}
                disabled={loading}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface ModelOptionProps {
  model: ModelInfo;
  isActive: boolean;
  onSelect: (name: string) => void;
  disabled: boolean;
}

function ModelOption({ model, isActive, onSelect, disabled }: ModelOptionProps) {
  return (
    <button
      className={`model-option ${isActive ? 'model-option-active' : ''}`}
      onClick={() => !isActive && onSelect(model.name)}
      disabled={disabled || !model.available}
    >
      <div className="model-option-header">
        <div className="model-option-info">
          <span className="model-option-name">{model.name}</span>
          <span className="model-option-size">
            {model.size_gb.toFixed(1)} GB · {model.device.toUpperCase()}
            {model.accuracy && ` · ${(model.accuracy * 100).toFixed(0)}% accuracy`}
          </span>
        </div>
        <div className="model-option-badges">
          {isActive && <span className="model-badge model-badge-active">✓ Active</span>}
          {model.recommended && !isActive && (
            <span className="model-badge model-badge-recommended">★ Recommended</span>
          )}
        </div>
      </div>
      {model.description && (
        <p className="model-option-description">{model.description}</p>
      )}
    </button>
  );
}

