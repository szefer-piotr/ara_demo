// Custom hook for handling data uploads - encapsulates file reading and processing logic
import { useState, useCallback } from 'react';
import { useAnalysisStore } from '../stores/analysisStore';
import { apiService } from '../services/api';
import { DataUploadResult } from '../types/data';
import Papa from 'papaparse';

export const useDataUpload = () => {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  
  const { 
    setCurrentData, 
    setColumnSummaries, 
    addFileId, 
    clearAllData 
  } = useAnalysisStore();

  // Helper function to determine if a column is numeric
  const isNumericColumn = (data: any[], columnName: string): boolean => {
    if (!data || data.length === 0) return false;
    return data.slice(0, 10).every(row => {
      const value = row[columnName];
      return value === null || value === undefined || value === '' || !isNaN(Number(value));
    });
  };

  // Parse CSV file locally
  const parseCSVFile = (file: File): Promise<any[]> => {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          if (results.errors.length > 0) {
            reject(new Error(`CSV parsing failed: ${results.errors[0].message}`));
          } else {
            resolve(results.data as any[]);
          }
        },
        error: (error) => {
          reject(new Error(`CSV parsing failed: ${error.message}`));
        }
      });
    });
  };

  // Handle CSV file reading and processing
  const handleFileUpload = useCallback(async (file: File): Promise<DataUploadResult> => {
    try {
      setIsUploading(true);
      setError(null);
      setProgress(0);

      // Validate file type
      if (!file.name.endsWith('.csv')) {
        throw new Error('Please upload a CSV file');
      }

      setProgress(15);

      // Parse CSV data locally first
      const csvData = await parseCSVFile(file);
      setProgress(35);

      // Upload and analyze the file (with error handling for missing backend)
      let upload, analysis;
      try {
        const uploadResponse = await apiService.uploadAndAnalyzeFile(file);
        upload = uploadResponse.upload;
        analysis = uploadResponse.analysis;
      } catch (apiError) {
        console.warn('Backend API not available, using local processing only:', apiError);
        // Fallback: create mock upload and analysis if API fails
        upload = { file_id: 'local_' + Date.now().toString(), message: 'Local processing' };
        analysis = { columns: [], summary: { row_count: 0, column_count: 0 } };
      }
      
      setProgress(85);

      // Update global state - use locally parsed data and backend column analysis
      const finalData = csvData || [];
      
      if (finalData.length === 0) {
        throw new Error('No data could be parsed from the CSV file');
      }
      
      setCurrentData(finalData);
      
      // Use backend column summaries if available, otherwise create basic ones from CSV data
      if (analysis.columns && analysis.columns.length > 0) {
        setColumnSummaries(analysis.columns);
      } else {
        // Create basic column summaries from the parsed CSV data
        const columnSummaries = Object.keys(finalData[0] || {}).map((columnName) => ({
          column_name: columnName,
          description: `Column ${columnName}`,
          type: isNumericColumn(finalData, columnName) ? 'numeric' : 'text',
          unique_value_count: new Set(finalData.map(row => row[columnName])).size
        }));
        setColumnSummaries(columnSummaries);
      }
      
      addFileId(upload.file_id);

      setProgress(100);

      // Get the column summaries that were set
      const currentColumnSummaries = analysis.columns && analysis.columns.length > 0 
        ? analysis.columns 
        : Object.keys(finalData[0] || {}).map((columnName) => ({
            column_name: columnName,
            description: `Column ${columnName}`,
            type: isNumericColumn(finalData, columnName) ? 'numeric' : 'text',
            unique_value_count: new Set(finalData.map(row => row[columnName])).size
          }));

      return {
        success: true,
        data: finalData,
        column_summaries: currentColumnSummaries,
        file_info: {
          rows: finalData.length,
          columns: Object.keys(finalData[0] || {}).length || 0,
          encoding: 'utf-8',
          delimiter: ','
        }
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      return {
        success: false,
        error: errorMessage
      };
    } finally {
      setIsUploading(false);
      setProgress(0);
    }
  }, [setCurrentData, setColumnSummaries, addFileId]);

  // Clear all uploaded data
  const handleClearData = useCallback(() => {
    clearAllData();
    setError(null);
  }, [clearAllData]);

  return {
    isUploading,
    error,
    progress,
    handleFileUpload,
    handleClearData
  };
};
