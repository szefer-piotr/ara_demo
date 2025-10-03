// Analysis Plan page - handles hypothesis planning and execution
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysisStore } from '../stores/analysisStore';
import { Card } from '../components/UI/Card';
import { Button } from '../components/UI/Button';
import { BarChart3, ArrowLeft, Play } from 'lucide-react';

const AnalysisPlan: React.FC = () => {
  const navigate = useNavigate();
  const { analyses, current_data } = useAnalysisStore();

  if (!current_data || analyses.length === 0) {
    navigate('/data');
    return null;
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          📊 Step 2: Analysis Plan
        </h1>
        <p className="text-gray-600">
          Review and execute the analysis plan for your hypotheses.
        </p>
      </div>

      {/* Hypotheses List */}
      <div className="mb-8">
        {analyses.map((hypothesis) => (
          <Card key={hypothesis.hypothesis_id} className="mb-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  {hypothesis.title}
                </h3>
                {hypothesis.plan_accepted ? (
                  <span className="text-sm text-green-600 font-medium">
                    ✓ Plan accepted
                  </span>
                ) : (
                  <span className="text-sm text-yellow-600 font-medium">
                    ⚪ Plan pending review
                  </span>
                )}
              </div>
              
              {hypothesis.analysis_plan.length > 0 ? (
                <div className="text-right">
                  <p className="text-sm text-gray-500">
                    {hypothesis.analysis_plan.length} steps planned
                  </p>
                  <Button size="sm" className="mt-1">
                    <Play className="w-4 h-4 mr-1" />
                    Execute
                  </Button>
                </div>
              ) : (
                <Button variant="outline" size="sm">
                  Generate Plan
                </Button>
              )}
            </div>
          </Card>
        ))}
      </div>

      {/* Navigation */}
      <div className="flex justify-between items-center">
        <Button
          variant="outline"
          onClick={() => navigate('/data')}
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to Data & Hypotheses
        </Button>
        
        <div className="flex gap-3">
          <Button variant="outline">
            Preview Report
          </Button>
          
          <Button 
            onClick={() => navigate('/report')}
            className="bg-green-600 hover:bg-green-700"
          >
            Generate Final Report →
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPlan;
