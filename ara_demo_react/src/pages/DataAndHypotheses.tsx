// Data and Hypotheses page - main data upload and hypothesis management
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysisStore } from '../stores/analysisStore';
import { FileUpload } from '../components/DataUpload/FileUpload';
import { DataPreview } from '../components/DataUpload/DataPreview';
import { HypothesisForm } from '../components/Hypotheses/HypothesisForm';
import { HypothesisList } from '../components/Hypotheses/HypothesisList';
import { Button } from '../components/UI/Button';
import { Card } from '../components/UI/Card';
import { CheckCircle, Upload, FileText } from 'lucide-react';

const DataAndHypotheses: React.FC = () => {
  const navigate = useNavigate();
  const { current_data, analyses } = useAnalysisStore();
  const [showUpload, setShowUpload] = useState(!current_data);

  // State management from session state equivalent  
  const isReady = current_data && analyses.length > 0;
  
  // Update showUpload state when current_data changes
  useEffect(() => {
    if (current_data && showUpload) {
      setShowUpload(false);
    }
  }, [current_data, showUpload]);

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          📊 Step 1: Upload Data & Add Hypotheses
        </h1>
        
        {/* Instructions Card - matches your yellow info box */}
        <Card className="bg-yellow-50 border-yellow-200" padding="lg">
          <h4 className="font-semibold text-yellow-900 mb-3">
            How to proceed:
          </h4>
          <ul className="text-sm text-yellow-700 space-y-1 mb-4">
            <li className="flex items-start">
              <Upload className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
              Upload your <strong>CSV file</strong>. The app will automatically summarize your dataset.
            </li>
            <li className="flex items-start">
              <FileText className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
              Edit column descriptions if needed. <strong>Be sure to provide accurate variable type</strong>. For example, is it an integer, continuous or a categorical variable.
            </li>
            <li className="flex items-start">
              <CheckCircle className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
              Add one or more <strong>hypotheses</strong> — type them in or import from a <code>.txt</code> file.
            </li>
          </ul>
          <div className="bg-yellow-100 border border-yellow-300 rounded p-2 text-sm text-yellow-800">
            ✅ Once your data is uploaded <strong>and</strong> at least one hypothesis is added, click <em>"Go to analysis plan"</em> to continue.
          </div>
        </Card>
      </div>

      {/* Data Upload Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <Upload className="w-5 h-5 mr-2" />
          Upload your data
        </h2>
        
        {showUpload ? (
          <FileUpload onSuccess={() => setShowUpload(false)} />
        ) : (
          <DataPreview onEdit={() => setShowUpload(true)} />
        )}
      </div>

      {/* Hypotheses Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <FileText className="w-5 h-5 mr-2" />
          Add hypotheses
        </h2>
        <HypothesisForm />
      </div>

      {/* Current Hypotheses Display */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Your hypotheses
        </h2>
        <HypothesisList />
        
        {analyses.length === 0 && (
          <Card className="text-center py-6">
            <FileText className="mx-auto w-12 h-12 text-gray-400 mb-2" />
            <p className="text-gray-500">
              None yet – add at least one hypothesis to continue.
            </p>
          </Card>
        )}
      </div>

      {/* Navigation Section - matches your Streamlit's ready check */}
      <div className="flex justify-between items-center border-t pt-6">
        <Button
          variant="outline"
          onClick={() => navigate('/')}
        >
          ← Back to Welcome
        </Button>
        
        {isReady ? (
          <Card className="bg-green-50 border-green-200 p-4 flex items-center">
            <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
            <div className="mr-4">
              <p className="font-medium text-green-800">Ready to continue</p>
              <p className="text-sm text-green-700">
                Data and {analyses.length} hypothesis{analyses.length !== 1 ? 'es' : ''} loaded
              </p>
            </div>
            <Button
              onClick={() => navigate('/analysis')}
              className="bg-green-600 hover:bg-green-700"
            >
              Go to Analysis Plan →
            </Button>
          </Card>
        ) : (
          <div className="text-sm text-gray-500">
            {!current_data && 'Upload data and '}
            {analyses.length === 0 && 'add at least one hypothesis '}
            to continue.
          </div>
        )}
      </div>
    </div>
  );
};

export default DataAndHypotheses;
