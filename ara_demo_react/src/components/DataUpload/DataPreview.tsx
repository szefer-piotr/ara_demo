// Data preview component - displays uploaded dataset and management actions
import React from 'react';
import { useAnalysisStore } from '../../stores/analysisStore';
import { Card } from '../UI/Card';
import { Button } from '../UI/Button';
import { Table, FileX, Edit, CheckCircle } from 'lucide-react';

interface DataPreviewProps {
  onEdit: () => void;
}

export const DataPreview: React.FC<DataPreviewProps> = ({ onEdit }) => {
  const { 
    current_data, 
    column_summaries, 
    clearAllData 
  } = useAnalysisStore();

  const handleRemoveFile = () => {
    clearAllData();
  };

  const dataArray = Array.isArray(current_data) ? current_data : [];
  const previewRows = dataArray.slice(0, 5); // Show first 5 rows
  const columnNames = dataArray.length > 0 ? Object.keys(dataArray[0]) : [];

  return (
    <div className="space-y-6">
      {/* File Info Header */}
      <div className="flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded-lg">
        <div className="flex items-center">
          <CheckCircle className="w-6 h-6 text-green-600 mr-3" />
          <div>
            <h3 className="font-semibold text-green-900">Dataset Loaded</h3>
            <p className="text-sm text-green-700">
              {dataArray.length.toLocaleString()} rows × {columnNames.length} columns
            </p>
          </div>
        </div>
        
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={onEdit}
            className="flex items-center"
          >
            <Edit className="w-4 h-4 mr-1" />
            Edit
          </Button>
          <Button
            variant="danger"
            onClick={handleRemoveFile}
            className="flex items-center"
          >
            <FileX className="w-4 h-4 mr-1" />
            Remove
          </Button>
        </div>
      </div>

      {/* Column Summaries */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Dataset Summary
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Edit column descriptions below if needed. Provide accurate variable types.
        </p>
        
        {column_summaries.length > 0 ? (
          <div className="space-y-3">
            {column_summaries.map((col, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-3">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium text-gray-900">{col.column_name}</h4>
                    <p className="text-sm text-gray-600">{col.description}</p>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-medium text-gray-700">
                      {col.type}
                    </span>
                    <p className="text-xs text-gray-500">
                      {col.unique_value_count} unique values
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-4 text-gray-500">
            Column summaries will appear here after data analysis...
          </div>
        )}
      </Card>

      {/* Data Table Preview */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Data Preview
        </h3>
        
        {previewRows.length > 0 && columnNames.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {columnNames.slice(0, 10).map((name, index) => (
                    <th
                      key={index}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {name}
                    </th>
                  ))}
                  {columnNames.length > 10 && (
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      ...
                    </th>
                  )}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {previewRows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {columnNames.slice(0, 10).map((column, colIndex) => (
                      <td
                        key={colIndex}
                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                      >
                        {typeof row[column] === 'object' 
                          ? JSON.stringify(row[column])
                          : String(row[column] ?? '')
                        }
                      </td>
                    ))}
                    {columnNames.length > 10 && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        ...
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            No data available for preview
          </div>
        )}
      </Card>
    </div>
  );
};
