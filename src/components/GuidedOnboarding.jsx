import React from 'react';
import {
  Upload,
  Camera,
  User,
  Brain,
  Activity,
  ShieldCheck,
  FileText,
  Stethoscope,
  ChevronRight,
  Check,
  Info,
  Sparkles,
  Zap,
  Eye,
  ArrowRight,
} from 'lucide-react';

// ─────────────────────────────────────────────────────────
// STEP DEFINITIONS
// ─────────────────────────────────────────────────────────
const STEPS = [
  {
    id: 1,
    title: 'Upload Scan',
    shortTitle: 'Scan',
    icon: Upload,
    color: 'from-cyan-400 to-blue-500',
    description: 'Upload or capture a dermatoscopic image of the skin lesion',
  },
  {
    id: 2,
    title: 'Patient Info',
    shortTitle: 'Patient',
    icon: User,
    color: 'from-violet-400 to-purple-500',
    description: 'Enter patient details to improve diagnostic accuracy',
  },
  {
    id: 3,
    title: 'AI Analysis',
    shortTitle: 'Analysis',
    icon: Brain,
    color: 'from-amber-400 to-orange-500',
    description: 'Our AI processes the image through a multi-stage pipeline',
  },
  {
    id: 4,
    title: 'Risk Score',
    shortTitle: 'Risk',
    icon: Activity,
    color: 'from-emerald-400 to-green-500',
    description: 'View the 1-10 malignancy risk score and 4-tier classification',
  },
  {
    id: 5,
    title: 'Doctor Referral',
    shortTitle: 'Doctor',
    icon: Stethoscope,
    color: 'from-rose-400 to-red-500',
    description: 'Get matched with the right specialist based on findings',
  },
  {
    id: 6,
    title: 'Full Report',
    shortTitle: 'Report',
    icon: FileText,
    color: 'from-sky-400 to-indigo-500',
    description: 'Download a printable clinical EHR report',
  },
];

// ─────────────────────────────────────────────────────────
// PIPELINE ANIMATION STAGES (shown during step 3)
// ─────────────────────────────────────────────────────────
const PIPELINE_STAGES = [
  {
    label: 'DullRazor Hair Removal',
    detail: 'Removing hair artifacts using morphological BlackHat transform & Telea inpainting',
    duration: 800,
    icon: '🪒',
  },
  {
    label: 'Color Constancy Normalization',
    detail: 'Adjusting lighting with Gray-World algorithm to remove camera flash bias',
    duration: 600,
    icon: '🎨',
  },
  {
    label: 'DenseNet121 Feature Extraction',
    detail: 'Extracting 1024-dimensional visual features using pre-trained deep neural network',
    duration: 1000,
    icon: '🧠',
  },
  {
    label: 'Patient Metadata Fusion',
    detail: 'Combining image features with age, sex & anatomical site through MLP branch',
    duration: 500,
    icon: '🔗',
  },
  {
    label: 'Focal Loss Classification',
    detail: 'Running 7-class prediction with class-imbalance-aware Focal Loss (γ=2.0)',
    duration: 700,
    icon: '🎯',
  },
  {
    label: 'Grad-CAM Explainability Map',
    detail: 'Generating neural attention heatmap showing which regions the AI focused on',
    duration: 600,
    icon: '🔥',
  },
  {
    label: 'Risk Stratification & Doctor Routing',
    detail: 'Computing 1-10 spectral risk score and matching to the right specialist',
    duration: 500,
    icon: '⚕️',
  },
];

// ─────────────────────────────────────────────────────────
// TOOLTIP COMPONENT
// ─────────────────────────────────────────────────────────
function Tooltip({ text, children }) {
  const [show, setShow] = React.useState(false);
  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded-xl bg-slate-800 border border-derma-border text-[11px] text-slate-200 whitespace-nowrap shadow-xl z-50 animate-in fade-in slide-in-from-bottom-1 duration-150">
          {text}
          <span className="absolute top-full left-1/2 -translate-x-1/2 -mt-px border-4 border-transparent border-t-slate-800" />
        </span>
      )}
    </span>
  );
}

// ─────────────────────────────────────────────────────────
// STEP PROGRESS BAR
// ─────────────────────────────────────────────────────────
export function StepProgressBar({ currentStep, completedSteps, onStepClick }) {
  return (
    <div className="w-full">
      {/* Desktop Progress */}
      <div className="hidden md:flex items-center justify-between relative">
        {/* Connecting line */}
        <div className="absolute top-5 left-8 right-8 h-0.5 bg-derma-border/60" />
        <div
          className="absolute top-5 left-8 h-0.5 bg-gradient-to-r from-derma-accent to-derma-teal transition-all duration-700 ease-out"
          style={{
            width: `${Math.max(0, ((Math.min(currentStep, STEPS.length) - 1) / (STEPS.length - 1)) * 100)}%`,
            maxWidth: 'calc(100% - 64px)',
          }}
        />

        {STEPS.map((step) => {
          const Icon = step.icon;
          const isActive = currentStep === step.id;
          const isCompleted = completedSteps.includes(step.id);
          const isAccessible = step.id <= Math.max(...completedSteps, 0) + 1;

          return (
            <button
              key={step.id}
              onClick={() => isAccessible && onStepClick(step.id)}
              disabled={!isAccessible}
              className={`relative z-10 flex flex-col items-center space-y-2 group transition-all duration-300 ${
                isAccessible ? 'cursor-pointer' : 'cursor-not-allowed opacity-40'
              }`}
            >
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-500 ${
                  isCompleted
                    ? 'bg-derma-teal text-slate-900 shadow-lg shadow-derma-teal/30 scale-100'
                    : isActive
                      ? `bg-gradient-to-br ${step.color} text-white shadow-lg shadow-derma-accent/30 scale-110 ring-4 ring-derma-accent/20`
                      : 'bg-derma-card border-2 border-derma-border text-slate-400'
                }`}
              >
                {isCompleted ? <Check className="w-5 h-5" /> : <Icon className="w-4 h-4" />}
              </div>
              <span
                className={`text-[11px] font-semibold transition-colors ${
                  isActive ? 'text-white' : isCompleted ? 'text-derma-teal' : 'text-slate-500'
                }`}
              >
                {step.shortTitle}
              </span>
            </button>
          );
        })}
      </div>

      {/* Mobile: Compact Progress Dots */}
      <div className="md:hidden flex items-center justify-center space-x-2 pb-3">
        {STEPS.map((step) => {
          const isActive = currentStep === step.id;
          const isCompleted = completedSteps.includes(step.id);
          return (
            <div
              key={step.id}
              className={`h-2 rounded-full transition-all duration-500 ${
                isActive
                  ? 'w-8 bg-derma-accent'
                  : isCompleted
                    ? 'w-2 bg-derma-teal'
                    : 'w-2 bg-derma-border'
              }`}
            />
          );
        })}
      </div>

      {/* Current Step Description Banner */}
      {currentStep >= 1 && currentStep <= STEPS.length && (
        <div className="mt-4 flex items-center justify-center space-x-2 text-center">
          <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-derma-surface/60 border border-derma-border/60">
            <span className="flex h-2 w-2 rounded-full bg-derma-accent animate-pulse" />
            <span className="text-xs font-semibold text-white">
              Step {currentStep} of {STEPS.length}:
            </span>
            <span className="text-xs text-slate-300">{STEPS[currentStep - 1].description}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// AI PROCESSING ANIMATION (Step 3)
// ─────────────────────────────────────────────────────────
export function AiProcessingAnimation({ onComplete }) {
  const [activeStage, setActiveStage] = React.useState(0);
  const [stageProgress, setStageProgress] = React.useState(0);
  const [completed, setCompleted] = React.useState(false);

  React.useEffect(() => {
    if (activeStage >= PIPELINE_STAGES.length) {
      setCompleted(true);
      const t = setTimeout(() => onComplete(), 600);
      return () => clearTimeout(t);
    }

    const stage = PIPELINE_STAGES[activeStage];
    const interval = 30;
    const steps = stage.duration / interval;
    let tick = 0;

    const timer = setInterval(() => {
      tick++;
      setStageProgress(Math.min(100, (tick / steps) * 100));
      if (tick >= steps) {
        clearInterval(timer);
        setTimeout(() => {
          setActiveStage((prev) => prev + 1);
          setStageProgress(0);
        }, 200);
      }
    }, interval);

    return () => clearInterval(timer);
  }, [activeStage, onComplete]);

  const overallProgress = ((activeStage + stageProgress / 100) / PIPELINE_STAGES.length) * 100;

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6 py-8">
      {/* Brain Animation Header */}
      <div className="text-center space-y-3">
        <div className="relative inline-flex items-center justify-center w-20 h-20 mx-auto">
          <div className="absolute inset-0 rounded-full bg-gradient-to-br from-derma-accent/20 to-derma-blue/20 animate-ping" />
          <div className="relative w-16 h-16 rounded-full bg-gradient-to-br from-derma-accent to-derma-blue flex items-center justify-center shadow-glow-cyan">
            <Brain className="w-8 h-8 text-white animate-pulse" />
          </div>
        </div>
        <h3 className="text-lg font-bold text-white">AI is Analyzing Your Scan...</h3>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          Processing through our multi-modal deep learning pipeline. Each stage enhances the
          diagnostic accuracy.
        </p>
      </div>

      {/* Overall Progress Bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
          <span>Pipeline Progress</span>
          <span>{Math.round(overallProgress)}%</span>
        </div>
        <div className="w-full h-2 rounded-full bg-derma-card border border-derma-border overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-derma-accent via-cyan-300 to-derma-teal transition-all duration-300 ease-out"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Pipeline Stages List */}
      <div className="space-y-2">
        {PIPELINE_STAGES.map((stage, idx) => {
          const isActive = idx === activeStage;
          const isDone = idx < activeStage;
          const isPending = idx > activeStage;

          return (
            <div
              key={idx}
              className={`flex items-center space-x-3 p-3 rounded-xl border transition-all duration-500 ${
                isActive
                  ? 'bg-derma-surface/80 border-derma-accent/50 shadow-lg shadow-derma-accent/10 scale-[1.02]'
                  : isDone
                    ? 'bg-derma-card/50 border-derma-teal/30'
                    : 'bg-derma-card/20 border-derma-border/30 opacity-40'
              }`}
            >
              {/* Status Icon */}
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-shrink-0 ${
                  isDone
                    ? 'bg-derma-teal/20 text-derma-teal'
                    : isActive
                      ? 'bg-derma-accent/20 text-derma-accent'
                      : 'bg-derma-card text-slate-500'
                }`}
              >
                {isDone ? <Check className="w-4 h-4" /> : <span>{stage.icon}</span>}
              </div>

              {/* Label and Detail */}
              <div className="flex-1 min-w-0">
                <p
                  className={`text-xs font-bold ${isDone ? 'text-derma-teal' : isActive ? 'text-white' : 'text-slate-500'}`}
                >
                  {stage.label}
                </p>
                {(isActive || isDone) && (
                  <p className="text-[10px] text-slate-400 mt-0.5 truncate">{stage.detail}</p>
                )}
              </div>

              {/* Progress indicator for active stage */}
              {isActive && (
                <div className="w-16 h-1.5 rounded-full bg-derma-card overflow-hidden flex-shrink-0">
                  <div
                    className="h-full rounded-full bg-derma-accent transition-all duration-100"
                    style={{ width: `${stageProgress}%` }}
                  />
                </div>
              )}

              {isDone && (
                <span className="text-[10px] text-derma-teal font-mono flex-shrink-0">Done</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Completed State */}
      {completed && (
        <div className="text-center pt-2 animate-in fade-in slide-in-from-bottom-2 duration-500">
          <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-derma-teal/20 border border-derma-teal/40 text-derma-teal text-xs font-bold">
            <Sparkles className="w-4 h-4" />
            <span>Analysis Complete — Generating Results...</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// "HOW IT WORKS" QUICK GUIDE
// ─────────────────────────────────────────────────────────
export function HowItWorksGuide({ onDismiss }) {
  return (
    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-derma-card via-derma-surface to-derma-card border border-derma-border p-6 sm:p-8">
      {/* Background decoration */}
      <div className="absolute top-0 right-0 w-64 h-64 rounded-full bg-derma-accent/5 blur-3xl" />
      <div className="absolute bottom-0 left-0 w-48 h-48 rounded-full bg-derma-blue/5 blur-3xl" />

      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-derma-accent to-derma-blue flex items-center justify-center">
              <Info className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">How SkinScan AI Works</h3>
              <p className="text-[11px] text-slate-400">Quick 60-second guide for new users</p>
            </div>
          </div>
          <button
            onClick={onDismiss}
            className="text-xs font-semibold text-derma-accent hover:text-white border border-derma-border hover:border-derma-accent px-3 py-1.5 rounded-lg transition-all"
          >
            Got it, start scanning →
          </button>
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            {
              step: 1,
              icon: Camera,
              title: 'Capture or Upload',
              desc: 'Take a photo of the skin spot using your phone camera, or upload an existing dermatoscopic image. You can also try our pre-loaded sample cases.',
              tip: 'Best results with close-up, well-lit photos',
            },
            {
              step: 2,
              icon: User,
              title: 'Add Patient Details',
              desc: "Enter the patient's age, sex, and where on the body the lesion is. This helps our AI be more accurate — like how a doctor would ask you questions.",
              tip: 'Age & location significantly affect risk',
            },
            {
              step: 3,
              icon: Brain,
              title: 'AI Analyzes the Image',
              desc: 'Our AI removes hair artifacts, normalizes colors, and runs the image through a DenseNet121 neural network trained on 10,015 clinical skin images.',
              tip: 'Takes about 4-5 seconds to complete',
            },
            {
              step: 4,
              icon: Activity,
              title: 'Get Your Risk Score',
              desc: "You'll see a clear 1-10 risk gauge (like a speedometer). Low scores (green) mean likely benign. High scores (red) mean further investigation is needed.",
              tip: 'Score ≥ 5.5 = "Investigate Further"',
            },
            {
              step: 5,
              icon: Stethoscope,
              title: 'Doctor Recommendation',
              desc: "Based on the risk level, we'll recommend the right type of doctor — from your local family doctor for normal moles, to a surgical oncologist for concerning findings.",
              tip: 'Cancer → Oncologist, Normal → Local Doctor',
            },
            {
              step: 6,
              icon: FileText,
              title: 'Download Report',
              desc: 'Generate a professional clinical report with all findings, AI visualizations, and doctor referral details. Print it or share with your healthcare provider.',
              tip: 'PDF-ready, print-optimized format',
            },
          ].map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.step}
                className="group p-4 rounded-2xl bg-derma-bg/60 border border-derma-border/60 hover:border-derma-accent/40 transition-all hover:shadow-lg hover:shadow-derma-accent/5"
              >
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-derma-surface flex items-center justify-center text-derma-accent font-bold text-sm border border-derma-border">
                    {item.step}
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <Icon className="w-4 h-4 text-derma-accent" />
                    <span className="text-sm font-bold text-white">{item.title}</span>
                  </div>
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed">{item.desc}</p>
                <div className="mt-2 flex items-center space-x-1 text-[10px] text-derma-accent/80">
                  <Zap className="w-3 h-3" />
                  <span className="font-medium">{item.tip}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// SECTION EXPLAINER (contextual heading for each results section)
// ─────────────────────────────────────────────────────────
export function SectionExplainer({ icon: Icon, title, subtitle, learnMore }) {
  const [showDetail, setShowDetail] = React.useState(false);

  return (
    <div className="flex items-start space-x-3 mb-4">
      <div className="w-9 h-9 rounded-xl bg-derma-surface border border-derma-border flex items-center justify-center flex-shrink-0 mt-0.5">
        <Icon className="w-4.5 h-4.5 text-derma-accent" />
      </div>
      <div className="flex-1">
        <div className="flex items-center space-x-2">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">{title}</h3>
          {learnMore && (
            <Tooltip text="Click to learn more">
              <button
                onClick={() => setShowDetail(!showDetail)}
                className="w-5 h-5 rounded-full bg-derma-surface border border-derma-border flex items-center justify-center hover:border-derma-accent transition-all"
              >
                <Info className="w-3 h-3 text-derma-muted" />
              </button>
            </Tooltip>
          )}
        </div>
        <p className="text-[11px] text-slate-400 mt-0.5">{subtitle}</p>
        {showDetail && learnMore && (
          <div className="mt-2 p-3 rounded-xl bg-derma-surface/50 border border-derma-border/60 text-[11px] text-slate-300 leading-relaxed animate-in fade-in slide-in-from-top-1 duration-200">
            {learnMore}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// STEP NAVIGATION BUTTONS
// ─────────────────────────────────────────────────────────
export function StepNavigation({ currentStep, onNext, onBack, nextLabel, nextDisabled, showBack }) {
  return (
    <div className="flex items-center justify-between pt-6">
      {showBack && currentStep > 1 ? (
        <button
          onClick={onBack}
          className="flex items-center space-x-2 px-5 py-2.5 text-xs font-semibold text-slate-300 bg-derma-card hover:bg-derma-surface border border-derma-border rounded-xl transition-all"
        >
          <ChevronRight className="w-4 h-4 rotate-180" />
          <span>Back</span>
        </button>
      ) : (
        <div />
      )}

      <button
        onClick={onNext}
        disabled={nextDisabled}
        className={`flex items-center space-x-2 px-6 py-2.5 text-xs font-bold uppercase tracking-wider rounded-xl transition-all transform active:scale-95 ${
          nextDisabled
            ? 'bg-derma-border text-slate-500 cursor-not-allowed'
            : 'bg-gradient-to-r from-derma-accent via-cyan-300 to-derma-blue text-slate-950 shadow-glow-cyan hover:opacity-95'
        }`}
      >
        <span>{nextLabel || 'Continue'}</span>
        <ArrowRight className="w-4 h-4" />
      </button>
    </div>
  );
}

export { STEPS, PIPELINE_STAGES, Tooltip };
