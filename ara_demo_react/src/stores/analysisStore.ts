// Global state management using Zustand
import { create } from 'zustand';
import { AnalysisState, Hypothesis, ColumnSummary } from '../types/data';

interface AnalysisStore extends AnalysisState {
  // Actions
  setCurrentData: (data: any[] | null) => void;
  setColumnSummaries: (summaries: ColumnSummary[]) => void;
  addHypothesis: (hypothesis: Hypothesis) => void;
  removeHypothesis: (id: string) => void;
  updateHypothesis: (id: string, updates: Partial<Hypothesis>) => void;
  setSelectedHypothesis: (id: string | null) => void;
  setSelectedStep: (id: string | null) => void;
  setEditMode: (mode: boolean) => void;
  addFileId: (fileId: string) => void;
  clearAllData: () => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisStore>((set, get) => ({
  // Initial state - matches Streamlit session state
  current_data: null,
  column_summaries: [],
  analyses: [],
  selected_hypothesis_id: null,
  selected_step_id: null,
  file_ids: [],
  edit_mode: false,

  // Actions
  setCurrentData: (data) => set({ current_data: data }),
  
  setColumnSummaries: (summaries) => set({ column_summaries: summaries }),
  
  addHypothesis: (hypothesis) => {
    set((state) => ({ 
      analyses: [...state.analyses, hypothesis],
      selected_hypothesis_id: hypothesis.hypothesis_id 
    }));
  },
  
  removeHypothesis: (id) => {
    const state = get();
    const updatedAnalyses = state.analyses.filter(a => a.hypothesis_id !== id);
    const newSelectedId = state.selected_hypothesis_id === id 
      ? updatedAnalyses.length > 0 
        ? updatedAnalyses[updatedAnalyses.length - 1].hypothesis_id 
        : null 
      : state.selected_hypothesis_id;
    set({ 
      analyses: updatedAnalyses,
      selected_hypothesis_id: newSelectedId 
    });
  },
  
  updateHypothesis: (id, updates) =>
    set((state) => ({
      analyses: state.analyses.map(a => 
        a.hypothesis_id === id ? { ...a, ...updates } : a
      )
    })),
  
  setSelectedHypothesis: (id) => set({ selected_hypothesis_id: id }),
  
  setSelectedStep: (id) => set({ selected_step_id: id }),
  
  setEditMode: (mode) => set({ edit_mode: mode }),
  
  addFileId: (fileId) => set((state) => ({ file_ids: [...state.file_ids, fileId] })),
  
  clearAllData: () => set({
    current_data: null,
    column_summaries: [],
    analyses: [],
    selected_hypothesis_id: null,
    selected_step_id: null,
    file_ids: [],
    edit_mode: false
  }),
  
  reset: () => get().clearAllData()
}));
