// Final Report page - generates and displays the scientific report
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysisStore } from '../stores/analysisStore';
import { Card } from '../components/UI/Card';
import { Button } from '../components/UI/Button';
import { FileCheck, ArrowLeft, Download } from 'lucide-react';

const FinalReport: React.FC = () => {
  const navigate = useNavigate();
  const { analyses, current_data } = useAnalysisStore();

  // Generate placeholder report based on hypotheses
  const generatePlaceholderReport = () => {
    return `
# Scientific Research Report

## Abstract

This report presents the results of ecological analyses conducted on ${current_data?.length || 0} data points.

## Methods

Statistical analyses were performed on the following hypotheses:

${analyses.map((h, i) => `${i + 1}. ${h.title}`).join('\n')}

## Results

${analyses.length > 0 
  ? `Analysis results for ${analyses.length} hypotheses will be presented here.`
  : 'No hypotheses available for analysis.'
}

## Discussion

Findings from the statistical analyses indicate patterns in the ecological data that warrant further investigation.

## References

Scientific literature referenced in this analysis will be listed here.
    `;
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              📋 Final Report
            </h1>
            <p className="text-gray-600 mt-1">
              Generated scientific report based on your analyses
            </p>
          </div>
          <Button 
            variant="outline"
            onClick={() => generatePlaceholderReport() && navigator.clipboard.writeText(generatePlaceholderReport())}
          >
            <Download className="w-4 h-4 mr-2" />
            Copy to Clipboard
          </Button>
        </div>
      </div>

      {/* Report Preview */}
      <Card className="min-h-screen">
        <div className="prose max-w-none">
          <div className="whitespace-pre-wrap">
            {generatePlaceholderReport()}
          </div>
        </div>
      </Card>

      {/* Actions */}
      <div className="flex gap-4 mt-8">
        <Button 
          variant="outline" 
          onClick={() => navigate('/analysis')}
          className="flex items-center"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Analysis
        </Button>
        <Button 
          className="flex items-center ml-auto"
          disabled
        >
          <FileCheck className="w-4 h-4 mr-2" />
          Complete Report
        </Button>
      </div>
    </div>
  );
};

export default FinalReport;
