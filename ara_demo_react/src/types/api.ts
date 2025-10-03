// API request and response type definitions

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface UploadResponse {
  file_id: string;
  message: string;
  file_version?: string;
}

export interface DataAnalysisResponse {
  columns: Array<{
    column_name: string;
    description: string;
    type: string;
    unique_value_count: number;
  }>;
  summary: {
    row_count: number;
    column_count: number;
  };
}

export interface AnalysisPlanResponse {
  steps: Array<{
    step_id: string;
    title: string;
    text: string;
  }>;
  plan_accepted: boolean;
}

export interface ExecutionResponse {
  results: any;
  message: string;
  files?: string[];
  generated_content?: {
    predictions?: any[];
    plots?: any[];
    tables?: any[];
  };
}

export interface ReportGenerationResponse {
  report: string;
  citation_count: number;
  has_images: boolean;
  processing_time: string;
}

// Request payloads
export interface FileUploadRequest {
  file: File;
  encoding?: string;
  delimiter?: string;
}

export interface HypothesisAddRequest {
  title: string;
  description?: string;
  data_summary?: any;
}

export interface AnalysisPlanRequest {
  hypothesis: string;
  data_summary: any;
}

export interface StepExecutionRequest {
  step_id: string;
  data: any;
  context?: any;
}
