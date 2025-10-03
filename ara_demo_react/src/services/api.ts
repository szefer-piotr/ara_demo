// API service layer - handles all communication with the Python backend
import axios from 'axios';
import { 
  ApiResponse, 
  UploadResponse, 
  DataAnalysisResponse,
  AnalysisPlanResponse,
  ExecutionResponse,
  ReportGenerationResponse 
} from '../types/api';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request/response interceptors for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Data upload and analysis
  uploadAndAnalyzeFile: async (file: File): Promise<{ upload: UploadResponse; analysis: DataAnalysisResponse }> => {
    const formData = new FormData();
    formData.append('file', file);
    
    // Upload file first
    const uploadResponse = await api.post<ApiResponse<UploadResponse>>('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    if (!uploadResponse.data.success || !uploadResponse.data.data) {
      throw new Error(uploadResponse.data.error || 'Upload failed');
    }
    
    // Analyze uploaded data
    const analysisResponse = await api.post<ApiResponse<DataAnalysisResponse>>('/analyze', {
      file_id: uploadResponse.data.data.file_id
    });
    
    if (!analysisResponse.data.success || !analysisResponse.data.data) {
      throw new Error(analysisResponse.data.error || 'Analysis failed');
    }
    
    return {
      upload: uploadResponse.data.data,
      analysis: analysisResponse.data.data
    };
  },

  // Generate analysis plan
  generateAnalysisPlan: async (hypothesis: string, dataSummary: any): Promise<AnalysisPlanResponse> => {
    const response = await api.post<ApiResponse<AnalysisPlanResponse>>('/generate-plan', {
      hypothesis,
      data_summary: dataSummary
    });
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Plan generation failed');
    }
    
    return response.data.data;
  },

  // Execute analysis step
  executeStep: async (stepId: string, stepData: any, context: any): Promise<ExecutionResponse> => {
    const response = await api.post<ApiResponse<ExecutionResponse>>('/execute-step', {
      step_id: stepId,
      data: stepData,
      context
    });
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Step execution failed');
    }
    
    return response.data.data;
  },

  // Generate final report
  generateReport: async (analyses: any[]): Promise<string> => {
    const response = await api.post<ApiResponse<ReportGenerationResponse>>('/generate-report', {
      analyses
    });
    
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Report generation failed');
    }
    
    return response.data.data.report;
  },

  // File management
  removeFile: async (fileId: string): Promise<void> => {
    const response = await api.delete(`/files/${fileId}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'File removal failed');
    }
  },

  // Health check
  healthCheck: async (): Promise<boolean> => {
    try {
      const response = await api.get('/health');
      return response.data?.status === 'ok';
    } catch {
      return false;
    }
  }
};
