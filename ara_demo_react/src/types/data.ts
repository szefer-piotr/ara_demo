// Type definitions for the research assistant application data structures

export interface ColumnSummary {
  column_name: string;
  description: string;
  type: string; // e.g., "categorical", "numeric", "text", "date"
  unique_value_count: number;
}

export interface DatasetSummary {
  columns: ColumnSummary[];
}

export interface AnalysisStep {
  step_id: string;
  title: string;
  text: string;
  finished?: boolean;
}

export interface AnalysisPlan {
  steps: AnalysisStep[];
}

export interface Hypothesis {
  hypothesis_id: string;
  title: string;
  data_summary?: DatasetSummary | null;
  analysis_plan: AnalysisStep[];
  plan_accepted?: boolean;
}

export interface AnalysisState {
  current_data: any[] | null;
  column_summaries: ColumnSummary[];
  analyses: Hypothesis[];
  selected_hypothesis_id: string | null;
  selected_step_id: string | null;
  file_ids: string[];
  edit_mode: boolean;
}

export interface DataUploadResult {
  success: boolean;
  data?: any[];
  column_summaries?: ColumnSummary[];
  error?: string;
  file_info?: {
    rows: number;
    columns: number;
    encoding: string;
    delimiter: string;
  };
}

export interface HypothesisFormData {
  title: string;
  description?: string;
}
