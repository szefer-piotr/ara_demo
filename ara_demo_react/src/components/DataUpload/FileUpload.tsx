// File upload component - handles CSV file upload with drag & drop
import React from 'react';
import { useDropzone } from 'react-dropzone';
import { useDataUpload } from '../../hooks/useDataUpload';
import { Button } from '../UI/Button';
import { Card } from '../UI/Card';
import { Upload, FileText, X } from 'lucide-react';
import toast from 'react-hot-toast';

interface FileUploadProps {
  onSuccess?: () => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onSuccess }) => {
  const { isUploading, error, progress, handleFileUpload } = useDataUpload();

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    
    try {
      const result = await handleFileUpload(file);
      
      if (result.success) {
        toast.success(`Successfully uploaded ${file.name}`);
        onSuccess?.();
      } else {
        toast.error(result.error || 'Upload failed');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      toast.error(errorMessage);
    }
  };

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/csv': ['.csv']
    },
    multiple: false,
    disabled: isUploading
  });

  return (
    <Card padding="lg" className="border-2 border-dashed border-gray-300 hover:border-gray-400 transition-colors">
      <div
        {...getRootProps()}
        className={`
          text-center cursor-pointer transition-all
          ${isDragActive ? 'bg-blue-50 border-blue-400' : 'hover:bg-gray-50'}
          ${isUploading ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
        `}
      >
        <input {...getInputProps()} />
        
        {isUploading ? (
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600"></div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                Processing your file...
              </h3>
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="mt-1 text-sm text-gray-600">{progress}% complete</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {isDragActive ? 'Drop your CSV file here' : 'Upload your CSV file'}
              </h3>
              <p className="mt-2 text-sm text-gray-600">
                Drag and drop your CSV file here, or click to browse
              </p>
              <p className="mt-1 text-xs text-gray-500">
                Supported format: CSV (.csv)
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center">
              <X className="h-5 w-5 text-red-400" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">
                  Upload Error
                </h3>
                <p className="mt-1 text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {fileRejections.length > 0 && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
            <h3 className="text-sm font-medium text-yellow-800">
              File Rejected
            </h3>
            <p className="mt-1 text-sm text-yellow-700">
              Please upload a valid CSV file (.csv format)
            </p>
          </div>
        )}
      </div>

      {!isUploading && (
        <div className="mt-4 text-center">
          <Button variant="primary" disabled={isUploading}>
            <FileText className="w-4 h-4 mr-2" />
            Browse Files
          </Button>
        </div>
      )}
    </Card>
  );
};
