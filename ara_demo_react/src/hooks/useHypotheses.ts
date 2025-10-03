// Custom hook for managing hypotheses - encapsulates all hypothesis CRUD operations
import { useCallback } from 'react';
import { useAnalysisStore } from '../stores/analysisStore';
import { Hypothesis, HypothesisFormData } from '../types/data';

export const useHypotheses = () => {
  const {
    analyses,
    selected_hypothesis_id,
    addHypothesis,
    removeHypothesis,
    updateHypothesis,
    setSelectedHypothesis
  } = useAnalysisStore();

  // Add a new hypothesis from form data
  const handleAddHypothesis = useCallback((formData: HypothesisFormData): string => {
    const newHypothesis: Hypothesis = {
      hypothesis_id: Math.random().toString(36).substr(2, 8), // Generate unique ID
      title: formData.title.trim(),
      analysis_plan: [],
      plan_accepted: false,
      data_summary: null
    };

    if (formData.description) {
      newHypothesis.data_summary = { description: formData.description };
    }

    addHypothesis(newHypothesis);
    return newHypothesis.hypothesis_id;
  }, [addHypothesis]);

  // Remove hypothesis by ID
  const handleRemoveHypothesis = useCallback((hypothesisId: string) => {
    removeHypothesis(hypothesisId);
    
    // If we removed the selected hypothesis, clear selection
    if (selected_hypothesis_id === hypothesisId) {
      const remainingAnalyses = analyses.filter(h => h.hypothesis_id !== hypothesisId);
      if (remainingAnalyses.length > 0) {
        setSelectedHypothesis(remainingAnalyses[remainingAnalyses.length - 1].hypothesis_id);
      } else {
        setSelectedHypothesis(null);
      }
    }
  }, [removeHypothesis, selected_hypothesis_id, setSelectedHypothesis, analyses]);

  // Update existing hypothesis
  const handleUpdateHypothesis = useCallback((
    hypothesisId: string, 
    updates: Partial<Hypothesis>
  ) => {
    updateHypothesis(hypothesisId, updates);
  }, [updateHypothesis]);

  // Import hypotheses from text file
  const handleImportHypotheses = useCallback((textContent: string): number => {
    const lines = textContent
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const newHypothesisIds: string[] = [];
    
    lines.forEach(line => {
      const newHypothesis: Hypothesis = {
        hypothesis_id: Math.random().toString(36).substr(2, 8),
        title: line,
        analysis_plan: [],
        plan_accepted: false
      };
      
      addHypothesis(newHypothesis);
      newHypothesisIds.push(newHypothesis.hypothesis_id);
    });

    // Select the last imported hypothesis
    if (newHypothesisIds.length > 0) {
      setSelectedHypothesis(newHypothesisIds[newHypothesisIds.length - 1]);
    }

    return lines.length;
  }, [addHypothesis, setSelectedHypothesis]);

  // Get current hypothesis data
  const getCurrentHypothesis = useCallback(() => {
    return analyses.find(h => h.hypothesis_id === selected_hypothesis_id);
  }, [analyses, selected_hypothesis_id]);

  return {
    // State
    hypotheses: analyses,
    currentHypothesis: getCurrentHypothesis(),
    
    // Actions
    addHypothesis: handleAddHypothesis,
    removeHypothesis: handleRemoveHypothesis,
    updateHypothesis: handleUpdateHypothesis,
    importHypotheses: handleImportHypotheses,
    selectHypothesis: setSelectedHypothesis
  };
};
