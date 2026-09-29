import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import DiagnosticWorkbench from './components/DiagnosticWorkbench';
import RiskGaugeCard from './components/RiskGaugeCard';
import GradCamVisualizer from './components/GradCamVisualizer';
import AbcdAnalysisCard from './components/AbcdAnalysisCard';
import ResearchMetricsDashboard from './components/ResearchMetricsDashboard';
import ClinicalReportModal from './components/ClinicalReportModal';
import MlPipelineModal from './components/MlPipelineModal';
import {
  StepProgressBar,
  AiProcessingAnimation,
  HowItWorksGuide,
  SectionExplainer,
  StepNavigation,
} from './components/GuidedOnboarding';
import { BENCHMARK_LESIONS } from './data/benchmarks';
import { runSkinScanInference } from './utils/imageProcessing';
import { analyzeWithBackend, checkBackendHealth, getBackendStatusLabel } from './utils/api';
import {
  ShieldCheck,
  Activity,
  Award,
  Heart,
  Sparkles,
  ExternalLink,
  Brain,
  Stethoscope,
  Eye,
  FileText,
  Info,
  ArrowRight,
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('triage'); // 'triage' | 'xai' | 'research'
  const [selectedBenchmark, setSelectedBenchmark] = useState(BENCHMARK_LESIONS[0]);
  const [customImageSrc, setCustomImageSrc] = useState(null);

  // Guided flow state
  const [currentStep, setCurrentStep] = useState(0); // 0 = how-it-works, 1-6 = wizard steps
  const [completedSteps, setCompletedSteps] = useState([]);
  const [showGuide, setShowGuide] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);

  // Patient Context Data
  const [patientData, setPatientData] = useState({
    age: 58,
    sex: 'male',
    localization: 'back',
    durationMonths: 7,
    historyOfMelanoma: true,
  });

  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [isPipelineModalOpen, setIsPipelineModalOpen] = useState(false);

  // Backend connection state
  const [backendHealth, setBackendHealth] = useState({
    online: false,
    modelLoaded: false,
    device: 'checking...',
    gpuName: null,
  });
  const [uploadedFile, setUploadedFile] = useState(null); // raw File object for backend

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth().then(setBackendHealth);
  }, []);

  // Mark a step as completed
  const completeStep = useCallback((stepId) => {
    setCompletedSteps((prev) => (prev.includes(stepId) ? prev : [...prev, stepId]));
  }, []);

  // Navigate to a step
  const goToStep = useCallback(
    (stepId) => {
      setCurrentStep(stepId);
      if (stepId >= 1) setShowGuide(false);
      // When going to triage steps, ensure triage tab is active
      if (stepId >= 1 && stepId <= 6) setActiveTab('triage');
    },
    [setActiveTab]
  );

  // Start the guided flow
  const handleStartGuided = useCallback(() => {
    setShowGuide(false);
    setCurrentStep(1);
    setActiveTab('triage');
  }, [setActiveTab]);

  // Handle the "How it works" dismiss
  const handleDismissGuide = useCallback(() => {
    handleStartGuided();
  }, [handleStartGuided]);

  // Update patient data
  const handleUpdatePatientData = (newData) => {
    setPatientData((prev) => {
      const updated = { ...prev, ...newData };
      // Dynamically re-calibrate risk score if we already have results
      if (analysisResult && currentStep >= 4) {
        const recalibrated = runSkinScanInference(updated, selectedBenchmark || analysisResult);
        setAnalysisResult(recalibrated);
      }
      return updated;
    });
  };

  // Select a benchmark case
  const handleSelectBenchmark = (lesion) => {
    setSelectedBenchmark(lesion);
    setCustomImageSrc(null);
    setUploadedFile(null);
    if (lesion) {
      setPatientData({
        age: lesion.metadata?.age || 50,
        sex: lesion.metadata?.sex || 'male',
        localization: lesion.metadata?.localization || 'back',
        durationMonths: lesion.metadata?.durationMonths || 6,
        historyOfMelanoma: lesion.metadata?.historyOfMelanoma || false,
      });
      // Auto-complete step 1 when a benchmark is selected
      completeStep(1);
    }
  };

  // Select custom image (uploaded file or webcam capture)
  const handleSelectCustomImage = (src, file) => {
    setCustomImageSrc(src);
    if (file) setUploadedFile(file);
    setSelectedBenchmark(null);
    completeStep(1);
  };

  // Run full multi-modal analysis (triggered from step 2 → step 3)
  const handleAnalyze = (uploadedSrc, file) => {
    setIsAnalyzing(true);
    if (uploadedSrc) setCustomImageSrc(uploadedSrc);
    if (file) setUploadedFile(file);
    completeStep(1);
    completeStep(2);
    setCurrentStep(3);
    setIsProcessing(true);
  };

  // Called when the AI processing animation finishes
  const handleProcessingComplete = useCallback(async () => {
    let result = null;

    // Try real backend first (if online and model loaded)
    if (backendHealth.online && backendHealth.modelLoaded && (uploadedFile || customImageSrc)) {
      const imageSource = uploadedFile || customImageSrc;
      const apiResult = await analyzeWithBackend(imageSource, patientData);
      if (apiResult.success) {
        result = apiResult.result;
        console.log('[SkinScan] ✅ Real ML inference from backend');
      } else {
        console.warn('[SkinScan] Backend failed, falling back to demo:', apiResult.error);
      }
    }

    // Fallback to client-side demo simulation
    if (!result) {
      result = runSkinScanInference(patientData, selectedBenchmark);
      result.isRealInference = false;
      console.log('[SkinScan] 🔄 Using demo simulation (backend offline)');
    }

    setAnalysisResult(result);
    setIsAnalyzing(false);
    setIsProcessing(false);
    completeStep(3);
    setCurrentStep(4); // Advance to Risk Score
  }, [patientData, selectedBenchmark, completeStep, backendHealth, uploadedFile, customImageSrc]);

  // Step navigation handlers
  const handleNextStep = useCallback(() => {
    const next = currentStep + 1;
    completeStep(currentStep);

    if (currentStep === 2) {
      // Step 2 → Step 3: trigger analysis
      handleAnalyze(customImageSrc);
      return;
    }

    if (currentStep === 6) {
      // Final step: open report
      setIsReportModalOpen(true);
      return;
    }

    setCurrentStep(Math.min(next, 6));
  }, [currentStep, completeStep, customImageSrc, handleAnalyze]);

  const handleBackStep = useCallback(() => {
    setCurrentStep((prev) => Math.max(1, prev - 1));
  }, []);

  // Quick-launch: skip guide for returning users
  const handleSkipToResults = useCallback(() => {
    setShowGuide(false);
    const res = runSkinScanInference(patientData, selectedBenchmark);
    setAnalysisResult(res);
    setCompletedSteps([1, 2, 3, 4, 5]);
    setCurrentStep(4);
    setActiveTab('triage');
  }, [patientData, selectedBenchmark]);

  return (
    <div className="min-h-screen bg-derma-bg text-slate-100 flex flex-col font-sans selection:bg-derma-accent/30 selection:text-white">
      {/* Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          // If switching to triage and no step set, go to step 1
          if (tab === 'triage' && currentStep === 0 && !showGuide) setCurrentStep(1);
        }}
        onOpenReport={() => setIsReportModalOpen(true)}
        onOpenPipeline={() => setIsPipelineModalOpen(true)}
        backendHealth={backendHealth}
      />

      {/* Main Content Area */}
      <main className="flex-1">
        {/* Hero Section */}
        <HeroSection
          onStartTriage={() => {
            if (showGuide) {
              handleStartGuided();
            } else {
              setActiveTab('triage');
              if (currentStep === 0) setCurrentStep(1);
            }
          }}
        />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
          {/* ═══════════════════════════════════════════════════════ */}
          {/* TAB 1: GUIDED TRIAGE FLOW                              */}
          {/* ═══════════════════════════════════════════════════════ */}
          {activeTab === 'triage' && (
            <div className="space-y-8">
              {/* "How It Works" Guide (shown first time only) */}
              {showGuide && (
                <div className="space-y-6">
                  <HowItWorksGuide onDismiss={handleDismissGuide} />

                  {/* Quick actions below the guide */}
                  <div className="flex flex-wrap items-center justify-center gap-3">
                    <button
                      onClick={handleStartGuided}
                      className="flex items-center space-x-2 px-6 py-3 text-sm font-bold text-slate-900 bg-gradient-to-r from-derma-accent via-cyan-300 to-derma-blue rounded-2xl shadow-glow-cyan hover:opacity-95 active:scale-95 transition-all"
                    >
                      <Sparkles className="w-4 h-4" />
                      <span>Start Step-by-Step Scan</span>
                    </button>
                    <button
                      onClick={handleSkipToResults}
                      className="flex items-center space-x-2 px-5 py-3 text-sm font-semibold text-slate-300 bg-derma-card border border-derma-border rounded-2xl hover:border-derma-accent transition-all"
                    >
                      <ArrowRight className="w-4 h-4" />
                      <span>Skip to Demo Results</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Step Progress Bar (visible once guide is dismissed) */}
              {!showGuide && currentStep >= 1 && (
                <StepProgressBar
                  currentStep={currentStep}
                  completedSteps={completedSteps}
                  onStepClick={(id) => {
                    // Allow clicking back to completed steps or next available
                    if (id <= Math.max(...completedSteps, 0) + 1) {
                      goToStep(id);
                    }
                  }}
                />
              )}

              {/* ─── STEP 1: Upload / Select Scan ─── */}
              {!showGuide && currentStep === 1 && (
                <div className="space-y-4">
                  <SectionExplainer
                    icon={Eye}
                    title="Step 1 — Upload or Select a Skin Image"
                    subtitle="Choose a pre-loaded clinical case to see how the AI works, or upload your own dermatoscopic photo."
                    learnMore="Our AI works best with close-up, well-lit images of skin lesions. Dermoscopic images (taken with a dermatoscope) provide the highest accuracy, but regular phone camera photos also work. You can try any of the 7 pre-loaded clinical benchmark cases from the HAM10000 dataset first."
                  />
                  <DiagnosticWorkbench
                    selectedBenchmark={selectedBenchmark}
                    onSelectBenchmark={handleSelectBenchmark}
                    customImageSrc={customImageSrc}
                    onSelectCustomImage={handleSelectCustomImage}
                    onAnalyze={(src, file) => {
                      if (src) setCustomImageSrc(src);
                      if (file) setUploadedFile(file);
                      completeStep(1);
                      goToStep(2);
                    }}
                    patientData={patientData}
                    onUpdatePatientData={handleUpdatePatientData}
                    isAnalyzing={isAnalyzing}
                    hidePatientForm={true}
                    hideAnalyzeButton={true}
                  />
                  <StepNavigation
                    currentStep={1}
                    onNext={() => {
                      completeStep(1);
                      goToStep(2);
                    }}
                    onBack={handleBackStep}
                    nextLabel="Continue to Patient Details"
                    nextDisabled={false}
                    showBack={false}
                  />
                </div>
              )}

              {/* ─── STEP 2: Patient Metadata ─── */}
              {!showGuide && currentStep === 2 && (
                <div className="space-y-4">
                  <SectionExplainer
                    icon={Info}
                    title="Step 2 — Enter Patient Information"
                    subtitle="Adding clinical context helps our AI make more accurate predictions. This mimics how a real doctor would assess you."
                    learnMore="Older patients (>60) have higher melanoma risk. Lesions on sun-exposed areas (back, face, scalp) are riskier. A family or personal history of melanoma significantly increases risk. These factors are weighted by our Multi-Modal MLP branch alongside the image analysis."
                  />
                  <DiagnosticWorkbench
                    selectedBenchmark={selectedBenchmark}
                    onSelectBenchmark={handleSelectBenchmark}
                    onAnalyze={handleAnalyze}
                    patientData={patientData}
                    onUpdatePatientData={handleUpdatePatientData}
                    isAnalyzing={isAnalyzing}
                    hideImageInput={true}
                  />
                  <StepNavigation
                    currentStep={2}
                    onNext={() => handleAnalyze(customImageSrc, uploadedFile)}
                    onBack={handleBackStep}
                    nextLabel="🧠 Run AI Analysis"
                    nextDisabled={false}
                    showBack={true}
                  />
                </div>
              )}

              {/* ─── STEP 3: AI Processing Animation ─── */}
              {!showGuide && currentStep === 3 && isProcessing && (
                <div className="space-y-4">
                  <AiProcessingAnimation onComplete={handleProcessingComplete} />
                </div>
              )}

              {/* ─── STEP 4: Risk Score Results ─── */}
              {!showGuide && currentStep >= 4 && analysisResult && (
                <div className="space-y-8">
                  {currentStep === 4 && (
                    <SectionExplainer
                      icon={Activity}
                      title="Step 4 — Your Risk Assessment Results"
                      subtitle="The gauge below shows the AI's malignancy risk score from 1 (safe) to 10 (critical). Below that, you'll see the 4-tier clinical classification."
                      learnMore={`The spectral score works like a traffic light: GREEN (1-3) = likely benign mole or wound, go to your local family doctor for routine check. YELLOW (3.5-5.5) = atypical, needs dermoscopic monitoring every 3-6 months. AMBER (5.5-7.5) = pre-malignant, see a dermatologist within 2-4 weeks. RED (7.5-10) = potential cancer, urgent referral to surgical oncologist within 48-72 hours.`}
                    />
                  )}

                  <RiskGaugeCard
                    result={analysisResult}
                    onOpenReport={() => setIsReportModalOpen(true)}
                  />

                  {currentStep === 4 && (
                    <StepNavigation
                      currentStep={4}
                      onNext={() => {
                        completeStep(4);
                        goToStep(5);
                      }}
                      onBack={handleBackStep}
                      nextLabel="View Doctor Recommendation"
                      showBack={true}
                    />
                  )}
                </div>
              )}

              {/* ─── STEP 5: Doctor Referral + XAI Visualizations ─── */}
              {!showGuide && currentStep >= 5 && analysisResult && (
                <div className="space-y-8">
                  {currentStep === 5 && (
                    <SectionExplainer
                      icon={Stethoscope}
                      title="Step 5 — Doctor Referral & AI Explanation"
                      subtitle="See which specialist doctor is recommended based on the findings, and how the AI arrived at its conclusion."
                      learnMore="The Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap shows which regions of the image the neural network focused on most. Red/hot areas = high attention (suspicious features). Blue/cool areas = low attention. The ABCD rule (Asymmetry, Border, Color, Diameter) is the standard clinical dermoscopy scoring method used by dermatologists worldwide."
                    />
                  )}

                  {/* Doctor referral info is already in the RiskGaugeCard above */}
                  {/* Show XAI explainability visualizations */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
                          <Brain className="w-4 h-4 text-derma-accent" />
                          <span>Grad-CAM: Where the AI Looked</span>
                        </h3>
                        <button
                          onClick={() => setActiveTab('xai')}
                          className="text-xs font-semibold text-derma-accent hover:underline flex items-center space-x-1"
                        >
                          <span>Full XAI Suite</span>
                          <ExternalLink className="w-3 h-3" />
                        </button>
                      </div>
                      <GradCamVisualizer result={analysisResult} customImageSrc={customImageSrc} />
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
                          <ShieldCheck className="w-4 h-4 text-derma-accent" />
                          <span>ABCD Clinical Scoring</span>
                        </h3>
                      </div>
                      <AbcdAnalysisCard result={analysisResult} />
                    </div>
                  </div>

                  {currentStep === 5 && (
                    <StepNavigation
                      currentStep={5}
                      onNext={() => {
                        completeStep(5);
                        goToStep(6);
                      }}
                      onBack={handleBackStep}
                      nextLabel="Generate Clinical Report"
                      showBack={true}
                    />
                  )}
                </div>
              )}

              {/* ─── STEP 6: Report Generation ─── */}
              {!showGuide && currentStep === 6 && analysisResult && (
                <div className="space-y-6">
                  <SectionExplainer
                    icon={FileText}
                    title="Step 6 — Your Clinical Report is Ready"
                    subtitle="Download or print a comprehensive EHR-format clinical report with all findings, scan images, AI visualizations, and doctor referral details."
                  />

                  {/* Report Preview Card */}
                  <div className="max-w-xl mx-auto p-8 rounded-3xl bg-gradient-to-br from-derma-card via-derma-surface to-derma-card border border-derma-border text-center space-y-6">
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-derma-accent to-derma-blue flex items-center justify-center shadow-glow-cyan">
                      <FileText className="w-8 h-8 text-white" />
                    </div>

                    <div>
                      <h3 className="text-lg font-bold text-white">Analysis Complete!</h3>
                      <p className="text-sm text-slate-400 mt-2 max-w-sm mx-auto">
                        All 7 pipeline stages have been processed. Your clinical triage report
                        includes:
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-left text-xs">
                      {[
                        '7-class disease classification',
                        '1-10 spectral risk score',
                        '4 multi-modal scan images',
                        'Grad-CAM attention heatmap',
                        'ABCD dermoscopy scoring',
                        'Specialist doctor referral',
                        'ICD-10 diagnostic code',
                        'Cancer vs. Normal wound triage',
                      ].map((item, idx) => (
                        <div key={idx} className="flex items-center space-x-2 text-slate-300">
                          <span className="w-1.5 h-1.5 rounded-full bg-derma-teal flex-shrink-0" />
                          <span>{item}</span>
                        </div>
                      ))}
                    </div>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
                      <button
                        onClick={() => {
                          completeStep(6);
                          setIsReportModalOpen(true);
                        }}
                        className="flex items-center space-x-2 px-6 py-3 text-sm font-bold text-slate-900 bg-gradient-to-r from-derma-accent via-cyan-300 to-derma-blue rounded-2xl shadow-glow-cyan hover:opacity-95 active:scale-95 transition-all"
                      >
                        <FileText className="w-4 h-4" />
                        <span>Open Full EHR Report</span>
                      </button>
                      <button
                        onClick={() => {
                          // Reset for a new scan
                          setCurrentStep(1);
                          setCompletedSteps([]);
                          setAnalysisResult(null);
                          setCustomImageSrc(null);
                          setSelectedBenchmark(BENCHMARK_LESIONS[0]);
                        }}
                        className="flex items-center space-x-2 px-5 py-3 text-sm font-semibold text-slate-300 bg-derma-card border border-derma-border rounded-2xl hover:border-derma-accent transition-all"
                      >
                        <Sparkles className="w-4 h-4" />
                        <span>Scan Another Lesion</span>
                      </button>
                    </div>
                  </div>

                  <StepNavigation
                    currentStep={6}
                    onNext={() => {
                      completeStep(6);
                      setIsReportModalOpen(true);
                    }}
                    onBack={handleBackStep}
                    nextLabel="Open Report"
                    showBack={true}
                  />
                </div>
              )}
            </div>
          )}

          {/* ═══════════════════════════════════════════════════════ */}
          {/* TAB 2: Explainable AI & ABCD Suite                     */}
          {/* ═══════════════════════════════════════════════════════ */}
          {activeTab === 'xai' && (
            <div className="space-y-10">
              {analysisResult ? (
                <>
                  <SectionExplainer
                    icon={Brain}
                    title="Explainable AI (XAI) Suite"
                    subtitle="Deep dive into how our neural network made its decision. Visualize attention maps, boundary contours, and clinical scoring."
                    learnMore="Grad-CAM generates heatmaps by computing gradients of the predicted class score flowing back into the final convolutional layer (conv5_block16_2_conv in DenseNet121). The ABCD scoring uses the Stolz formula: TDS = 1.3×A + 0.1×B + 0.5×C + 0.5×D. A TDS > 5.45 suggests malignancy."
                  />
                  <GradCamVisualizer result={analysisResult} customImageSrc={customImageSrc} />
                  <AbcdAnalysisCard result={analysisResult} />
                </>
              ) : (
                <div className="text-center py-16 space-y-4">
                  <Brain className="w-12 h-12 text-derma-muted mx-auto" />
                  <p className="text-sm text-slate-400">
                    Run an analysis first to see the XAI visualizations.
                  </p>
                  <button
                    onClick={() => {
                      setActiveTab('triage');
                      if (currentStep === 0) setCurrentStep(1);
                    }}
                    className="text-xs font-semibold text-derma-accent hover:underline"
                  >
                    Go to Point-of-Care Triage →
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ═══════════════════════════════════════════════════════ */}
          {/* TAB 3: Model Analytics & Research                      */}
          {/* ═══════════════════════════════════════════════════════ */}
          {activeTab === 'research' && (
            <div className="space-y-10">
              <SectionExplainer
                icon={ShieldCheck}
                title="Model Performance & Research Metrics"
                subtitle="Comprehensive validation metrics, confusion matrix, ROC-AUC curves, and alignment with UN Sustainable Development Goals."
                learnMore="These metrics are computed on the HAM10000 test split (15% holdout). The confusion matrix shows per-class performance, while the ROC curve demonstrates the model's ability to discriminate between all 7 lesion types. Our work aligns with SDG 3 (Good Health), SDG 9 (Innovation), and SDG 10 (Reduced Inequalities in healthcare access)."
              />
              <ResearchMetricsDashboard />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-derma-border/80 bg-derma-card/80 py-10 mt-16 text-xs text-derma-muted">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-white text-sm">SkinScan AI Triage System</span>
                <span className="px-2 py-0.5 rounded bg-derma-surface text-derma-teal font-mono text-[10px]">
                  v2.0 Hybrid DenseNet
                </span>
              </div>
              <p className="mt-1 text-[11px] text-slate-400">
                Major Project Phase-2 • Bachelor of Engineering in CSE, B.M.S. College of
                Engineering (Autonomous VTU)
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                Students: Akshat Kumar Sinha, Anmol Bhattarai, Ayaan Shrestha, Ayush Ranjan | Guide:
                Megavalli M
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-[11px]">
              <button
                onClick={() => setIsPipelineModalOpen(true)}
                className="hover:text-derma-accent transition-colors"
              >
                Model Architecture
              </button>
              <button
                onClick={() => setIsReportModalOpen(true)}
                className="hover:text-derma-accent transition-colors"
              >
                EHR Report Format
              </button>
              <a
                href="https://www.dermasensor.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-derma-accent transition-colors flex items-center space-x-1"
              >
                <span>DermaSensor Inspiration</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-derma-border/40 flex flex-col sm:flex-row items-center justify-between text-[10px] text-slate-500">
            <p>© 2026 SkinScan AI • Academic Research & Clinical Decision Support Protocol</p>
            <p className="mt-2 sm:mt-0 flex items-center space-x-1">
              <span>Service to Humanity is Service to God</span>
              <Award className="w-3.5 h-3.5 text-derma-accent" />
            </p>
          </div>
        </div>
      </footer>

      {/* EHR Clinical Report Modal */}
      <ClinicalReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        result={analysisResult}
        patientData={patientData}
        customImageSrc={customImageSrc}
        acquisitionMode={customImageSrc ? 'custom' : 'benchmark'}
      />

      {/* ML Pipeline & Architecture Modal */}
      <MlPipelineModal isOpen={isPipelineModalOpen} onClose={() => setIsPipelineModalOpen(false)} />
    </div>
  );
}
