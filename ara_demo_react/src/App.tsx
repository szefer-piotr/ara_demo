// Main App component - React Router setup and application entry point
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

// Pages
import Welcome from './pages/Welcome';
import DataAndHypotheses from './pages/DataAndHypotheses';
import AnalysisPlan from './pages/AnalysisPlan';
import FinalReport from './pages/FinalReport';

// Layout
import Layout from './components/Layout/Layout';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Layout>
          <Routes>
            <Route path="/" element={<Welcome />} />
            <Route path="/data" element={<DataAndHypotheses />} />
            <Route path="/analysis" element={<AnalysisPlan />} />
            <Route path="/report" element={<FinalReport />} />
          </Routes>
        </Layout>
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;
