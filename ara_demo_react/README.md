# React AI Research Assistant

This React application for ecological research analysis tool.

## 📚 Directory Structure & File Explanations

### Root Configuration Files
- **package.json** - Project dependencies and build scripts
- **vite.config.ts** - Vite build tool configuration with API proxy
- **tsconfig.json** - TypeScript compiler configuration
- **tailwind.config.js** - Tailwind CSS styling configuration
- **postcss.config.js** - PostCSS configuration for Tailwind
- **index.html** - HTML entry point
- **README.md** - This file explaining everything

### `/src` - Source Code Directory

#### Types & Data Structures
- **src/types/data.ts** - TypeScript interfaces matching your Pyfloric schemas (ColumnSummary, Hypothesis, etc.)
- **src/types/api.ts** - API request/response type definitions for backend communication

#### State Management 
- **src/stores/analysisStore.ts** - Zustand global state store (replaces Streamlit session_state)
- **src/hooks/useDataUpload.ts** - Custom hook for file upload logic
- **src/hooks/useHypotheses.ts** - Custom hook for hypothesis management

#### API Layer
- **src/services/api.ts** - HTTP client for backend Python API calls

#### Layout Components (`src/components/Layout/`)
- **Layout.tsx** - Main application layout with sidebar and content area
- **Sidebar.tsx** - Navigation sidebar that shows file status from session state
- **Header.tsx** - Top header bar

#### UI Components (`src/components/UI/`)
- **Button.tsx** - Reusable button with variants (primary, secondary, danger, outline)
- **Card.tsx** - Content container card component

#### Data Upload (`src/components/DataUpload/`)
- **FileUpload.tsx** - Drag & drop CSV file uploader (replaces Streamlit uploader)
- **DataPreview.tsx** - Shows uploaded data table and column summaries (matches df.head())

#### Hypothesis Components (`src/components/Hypotheses/`)
- **HypothesisForm.tsx** - Adding/importing hypotheses form (replaces hypothesis input)
- **HypothesisList.tsx** - Display and manage existing hypotheses (replaces H list with buttons)

#### Pages (`src/pages/`)
- **Welcome.tsx** - Landing page matching your Streamlit intro
- **DataAndHypotheses.tsx** - Main work page (equivalent to your 2_Data_and_Hypotheses.py)
- **AnalysisPlan.tsx** - Analysis planning and execution
- **FinalReport.tsx** - Report generation (equivalent to 4_Final_Report.py)

#### Styling & Entry Point
- **src/styles/globals.css** - Global CSS + Tailwind base
- **src/App.tsx** - Main app router setup (routes, Layout component)
- **src/main.tsx** - React entry point

## 🚀 How to Start

```bash
# 1. Install dependencies
npm install

# 2. Set up your backend API to handle these endpoints:
# - POST /api/upload - file upload
# - POST /api/analyze - data analysis  
# - POST /api/generate-plan - generate analysis plans
# - POST /api/execute-step - run analysis steps
# - POST /api/generate-report - create final reports

# 3. Start backend (your Python server) on port 8000
python -m backend.main  # or your Python server command

# 4. Start React development server
npm run dev

# Open http://localhost:3000
```

## ✨ What Each File Replaces from Your Streamlit App

### Direct Replacements
- `2_Data_and_Hypotheses.py` → `src/pages/DataAndHypotheses.tsx` 
- `1_Welcome.py` → `src/pages/Welcome.tsx`
- `3_Analysis_Plan.py` → `src/pages/AnalysisPlan.tsx`
- `4_Final_Report.py` → `src/pages/FinalReport.tsx`
- `sidebar.py` → `src/components/Layout/Sidebar.tsx`

### Split & Reorganized
- Session state management → `src/stores/analysisStore.ts`
- File operations → `src/components/DataUpload/*`
- Form logic → `src/components/Hypotheses/*`
- API calls → `src/services/api.ts`

### Added New Functionality
- Loading states
- Progress tracking
- Error handling
- Responsive design
- TypeScript type safety

This maintains 100% feature parity with your Streamlit app but with modern React architecture and better UX!
