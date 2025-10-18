/**
 * Settings View Component (Placeholder)
 */

import { Card, CardHeader, CardContent } from './Card';
import './SettingsView.css';

export function SettingsView() {
  return (
    <div className="settings-view">
      <div className="settings-header">
        <h2 className="settings-title">Settings</h2>
        <p className="settings-subtitle">Customize your experience</p>
      </div>

      <div className="settings-content">
        <Card>
          <CardHeader title="Coming Soon" subtitle="Settings panel is under development" />
          <CardContent>
            <div className="settings-placeholder">
              <div className="settings-icon">⚙️</div>
              <p className="settings-text">
                Settings and preferences will be available here soon:
              </p>
              <ul className="settings-list">
                <li>Theme selection (Light/Dark mode)</li>
                <li>Model parameters (temperature, max length)</li>
                <li>API endpoint configuration</li>
                <li>Language preferences</li>
                <li>Performance options</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

