// Hypothesis list component - displays, manages, and navigates hypotheses
import React from 'react';
import { useHypotheses } from '../../hooks/useHypotheses';
import { Button } from '../UI/Button';
import { Card } from '../UI/Card';
import { FileText, Trash2, Hash } from 'lucide-react';

export const HypothesisList: React.FC = () => {
  const { 
    hypotheses, 
    currentHypothesis, 
    removeHypothesis, 
    selectHypothesis 
  } = useHypotheses();

  const handleRemove = (hypothesisId: string, title: string) => {
    if (window.confirm(`Remove hypothesis: "${title}"?`)) {
      removeHypothesis(hypothesisId);
    }
  };

  if (hypotheses.length === 0) {
    return (
      <Card className="text-center py-8">
        <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          No hypotheses yet
        </h3>
        <p className="text-gray-500">
          Add at least one hypothesis above before proceeding.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          Your Hypotheses
        </h3>
        <span className="text-sm text-gray-500">
          Total: {hypotheses.length}
        </span>
      </div>

      <p className="text-sm text-gray-600">
        Your hypotheses will appear here. You can remove them or add new ones below.
        When you're ready, click "Go to Analysis Plan" to continue.
      </p>

      <div className="grid gap-3">
        {hypotheses.map((hypothesis, index) => (
          <Card
            key={hypothesis.hypothesis_id}
            className={`
              border-2 transition-all
              ${
                currentHypothesis?.hypothesis_id === hypothesis.hypothesis_id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }
            `}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center mb-2">
                  <Hash className="w-4 h-4 text-gray-400 mr-2 flex-shrink-0" />
                  <span className="text-xs font-mono text-gray-500">
                    ID: {hypothesis.hypothesis_id}
                  </span>
                </div>
                
                <h4 className="text-base font-medium text-gray-900 line-clamp-2">
                  {hypothesis.title}
                </h4>
                
                {hypothesis.data_summary && (
                  <div className="mt-2">
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                      Data Linked
                    </span>
                  </div>
                )}

                {/* Status indicators */}
                <div className="flex items-center mt-2 space-x-2">
                  {hypothesis.plan_accepted ? (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Plan Accepted
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                      Draft
                    </span>
                  )}

                  {hypothesis.analysis_plan.length > 0 && (
                    <span className="text-xs text-gray-500">
                      {hypothesis.analysis_plan.length} steps planned
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center space-x-2 ml-4">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => selectHypothesis(hypothesis.hypothesis_id)}
                  className="whitespace-nowrap"
                >
                  View
                </Button>
                
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => handleRemove(hypothesis.hypothesis_id, hypothesis.title)}
                  className="p-2"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Current Selection Info */}
      {currentHypothesis && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">
            Selected Hypothesis
          </h4>
          <p className="text-sm text-blue-700">
            {currentHypothesis.title}
          </p>
        </div>
      )}
    </div>
  );
};
