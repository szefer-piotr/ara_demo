// Hypothesis form component - handles adding and importing hypotheses
import React, { useState, useRef } from 'react';
import { useHypotheses } from '../../hooks/useHypotheses';
import { Button } from '../UI/Button';
import { Card } from '../UI/Card';
import { Plus, FileText, Upload } from 'lucide-react';
import toast from 'react-hot-toast';

export const HypothesisForm: React.FC = () => {
  const [typedText, setTypedText] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { addHypothesis, importHypotheses } = useHypotheses();

  const handleTypedSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!typedText.trim()) {
      toast.error('Please enter a hypothesis');
      return;
    }

    try {
      const id = addHypothesis({
        title: typedText.trim(),
        description: undefined
      });
      
      setTypedText('');
      toast.success('Hypothesis added successfully');
    } catch (err) {
      toast.error('Failed to add hypothesis');
    }
  };

  const handleFileImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsImporting(true);
    
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const text = event.target?.result as string;
        const lineCount = importHypotheses(text);
        
        if (lineCount > 0) {
          toast.success(`Successfully imported ${lineCount} hypotheses`);
        } else {
          toast.error('No valid hypotheses found in the file');
        }
      } catch (err) {
        toast.error('Failed to import hypotheses');
      } finally {
        setIsImporting(false);
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    };
    
    reader.readAsText(file);
  };

  return (
    <Card>
      <div className="space-y-6">
        {/* Manual Entry Form */}
        <form onSubmit={handleTypedSubmit} className="space-y-4">
          <div>
            <label 
              htmlFor="hypothesis-input" 
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Type a new hypothesis
            </label>
            <input
              id="hypothesis-input"
              type="text"
              value={typedText}
              onChange={(e) => setTypedText(e.target.value)}
              placeholder="Enter your hypothesis here..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={isImporting}
            />
          </div>
          
          <Button
            type="submit"
            disabled={!typedText.trim() || isImporting}
            className="flex items-center"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Hypothesis
          </Button>
        </form>

        {/* File Import */}
        <div className="border-t pt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">
            Or import from file
          </h3>
          
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt"
                onChange={handleFileImport}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                disabled={isImporting}
              />
              <p className="mt-1 text-xs text-gray-500">
                Upload a TXT file with one hypothesis per line
              </p>
            </div>
            
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={isImporting}
              className="flex items-center"
            >
              <Upload className="w-4 h-4 mr-2" />
              Browse Files
            </Button>
          </div>

          {isImporting && (
            <div className="mt-2 flex items-center text-sm text-blue-600">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
              Importing...
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
