// Sidebar component - navigation and current state display
import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAnalysisStore } from '../../stores/analysisStore';
import { Button } from '../UI/Button';
import { 
  Home, 
  Database, 
  BarChart3, 
  FileCheck,
  Upload,
  FileText 
} from 'lucide-react';

const navigationItems = [
  { 
    name: 'Welcome', 
    href: '/', 
    icon: Home,
    description: 'Getting started'
  },
  { 
    name: 'Data & Hypotheses', 
    href: '/data', 
    icon: Database,
    description: 'Upload and manage'
  },
  { 
    name: 'Analysis Plan', 
    href: '/analysis', 
    icon: BarChart3,
    description: 'Statistical analysis'
  },
  { 
    name: 'Final Report', 
    href: '/report', 
    icon: FileCheck,
    description: 'Generate results'
  },
];

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  const { 
    current_data, 
    analyses, 
    column_summaries,
    clearAllData 
  } = useAnalysisStore();

  // Check if we have any data
  const hasData = current_data && current_data.length > 0;
  const hypothesisCount = analyses.length;

  const isActive = (href: string) => location.pathname === href;

  const handleClearAll = () => {
    if (window.confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
      clearAllData();
      navigate('/');
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-900">
          Research Assistant
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Ecological Analysis Tool
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navigationItems.map((item) => (
            <li key={item.name}>
              <Button
                variant={isActive(item.href) ? "primary" : "outline"}
                onClick={() => navigate(item.href)}
                className="w-full justify-start mb-1"
              >
                <item.icon className="w-4 h-4 mr-3" />
                <div className="text-left">
                  <div className="font-medium">{item.name}</div>
                  <div className="text-xs text-gray-500">{item.description}</div>
                </div>
              </Button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Status Summary */}
      <div className="border-t border-gray-200 p-4 space-y-4">
        {/* Data Status */}
        <div>
          <h3 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
            <Upload className="w-4 h-4 mr-2" />
            Data
          </h3>
          {hasData ? (
            <div className="text-xs text-gray-600">
              <div className="font-medium text-green-700">
                ✓ {Array.isArray(current_data) ? current_data.length.toLocaleString() : 0} rows
              </div>
              <div className="text-gray-500">
                {column_summaries.length} columns analyzed
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-500">
              No dataset uploaded yet
            </div>
          )}
        </div>

        {/* Hypotheses Status */}
        <div>
          <h3 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
            <FileText className="w-4 h-4 mr-2" />
            Hypotheses
          </h3>
          {hypothesisCount > 0 ? (
            <div className="text-xs text-gray-600">
              <div className="font-medium text-green-700">
                ✓ {hypothesisCount} hypothesis{hypothesisCount !== 1 ? 'es' : ''}
              </div>
              <div className="text-gray-500">
                {analyses.filter(a => a.plan_accepted).length} accepted
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-500">
              No hypotheses yet
            </div>
          )}
        </div>

        {/* Clear All Button */}
        {(hasData || hypothesisCount > 0) && (
          <div className="pt-2">
            <Button
              variant="danger"
              size="sm"
              onClick={handleClearAll}
              className="w-full text-xs"
            >
              ↻ Clear All Data
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export { Sidebar };
