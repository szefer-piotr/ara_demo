// Welcome page - matches Streamlit welcome screen functionality
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/UI/Button';
import { Card } from '../components/UI/Card';
import { FileText, ArrowRight } from 'lucide-react';

const Welcome: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Welcome to Your Research Assistant
        </h1>
        
        <Card padding="lg" className="bg-yellow-50 border-yellow-200">
          <div className="flex items-start">
            <FileText className="w-8 h-8 text-yellow-600 mr-4 flex-shrink-0" />
            <div className="text-left">
              <h2 className="text-xl font-semibold text-yellow-900 mb-2">
                🧭 Research Assistant Guide
              </h2>
              <p className="text-yellow-700 mb-4">
                This app helps you analyze your dataset, test hypotheses, and generate a final report — all with AI assistance.
              </p>
              <ul className="text-sm text-yellow-700 space-y-1">
                <li>🎯 <strong>Upload datasets</strong> - CSV files with your research data</li>
                <li>🧪 <strong>Create hypotheses</strong> - Define what you want to test</li>
                <li>📊 <strong>Statistical analysis</strong> - Automated statistical workflows</li>
                <li>📝 <strong>Generate reports</strong> - Publication-ready scientific reports</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6">
            <Button
              onClick={() => navigate('/data')}
              className="bg-blue-600 hover:bg-blue-700"
            >
              Start Analysis
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </Card>
      </div>

      {/* Quick Start Instructions */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">1. Upload Data</h3>
          <p className="text-gray-600 text-sm">
            Upload your CSV dataset. The app will automatically analyze structure and variable types.
          </p>
        </Card>
        
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">2. Add Hypotheses</h3>
          <p className="text-gray-600 text-sm">
            Define what you want to test. Import multiple hypotheses from text files.
          </p>
        </Card>
        
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">3. Generate Report</h3>
          <p className="text-gray-600 text-sm">
            Get a complete scientific report with statistical results and interpretations.
          </p>
        </Card>
      </div>

      {/* Usage Tips */}
      <Card className="bg-blue-50 border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-2">💡 Tips</h3>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Come prepared with your research question clearly defined</li>
          <li>• Make sure your CSV has column headers and numerical data for key analyses</li>
          <li>• Provide accurate variable type information when prompted</li>
          <li>• The AI will help decide appropriate statistical tests based on your data</li>
        </ul>
      </Card>

      <div className="mt-8 text-center">
        <Button
          onClick={() => navigate('/data')}
          size="lg"
          className="mr-4"
        >
          Get Started
        </Button>
      </div>
    </div>
  );
};

export default Welcome;
