// Header component - top navigation and user info
import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Ecological Research Assistant
          </h2>
          <p className="text-sm text-gray-500">
            Analyze your data, test hypotheses, and generate reports
          </p>
        </div>
        
        {/* Add any header controls here in the future */}
        <div className="flex items-center space-x-4">
          {/* Status indicator could go here */}
        </div>
      </div>
    </header>
  );
};

