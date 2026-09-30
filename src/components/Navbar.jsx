import React from 'react';
import { ShieldCheck, Activity, Brain, FileText, Cpu, Sparkles, Home } from 'lucide-react';

export default function Navbar({
  activeTab,
  setActiveTab,
  onGoHome,
  onGoToTriage,
  isHome,
  onOpenReport,
  onOpenPipeline,
  backendHealth,
}) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-derma-border/80 bg-derma-bg/90 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 sm:h-20 gap-2">
          {/* Logo & Clinical Brand -> Click takes user to Home Page */}
          <div
            role="button"
            tabIndex={0}
            title="Go to Home Page"
            aria-label="SkinScan.AI Home Page"
            className="flex items-center space-x-2.5 sm:space-x-3 cursor-pointer min-w-0 flex-shrink-0 group hover:opacity-95 active:scale-95 transition-all select-none"
            onClick={onGoHome || (() => setActiveTab('triage'))}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (onGoHome) onGoHome();
                else setActiveTab('triage');
              }
            }}
          >
            <div className="relative flex items-center justify-center w-9 h-9 sm:w-11 sm:h-11 rounded-xl bg-gradient-to-tr from-derma-blue to-derma-accent p-0.5 shadow-glow-cyan flex-shrink-0 group-hover:shadow-[0_0_22px_rgba(34,211,238,0.55)] transition-all">
              <div className="w-full h-full bg-derma-card rounded-[10px] flex items-center justify-center">
                <Activity className="w-5 h-5 sm:w-6 sm:h-6 text-derma-accent group-hover:scale-110 transition-transform" />
              </div>
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 sm:w-3 sm:h-3 bg-derma-teal rounded-full animate-ping" />
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 sm:w-3 sm:h-3 bg-derma-teal rounded-full" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center space-x-2">
                <span className="text-lg sm:text-xl font-bold tracking-tight text-white font-sans whitespace-nowrap group-hover:text-derma-accent transition-colors">
                  SkinScan<span className="text-derma-accent font-extrabold">.AI</span>
                </span>
                <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-derma-surface text-derma-accent rounded border border-derma-border whitespace-nowrap">
                  DermaSensor UI
                </span>
              </div>
              <p className="hidden sm:block text-[11px] text-derma-muted font-medium truncate">
                4-Stage Clinical Risk Stratification & Triage
              </p>
            </div>
          </div>

          {/* Desktop Navigation Links (xl+) */}
          <nav className="hidden xl:flex items-center space-x-1 bg-derma-card/80 p-1.5 rounded-xl border border-derma-border flex-shrink-0">
            <button
              onClick={onGoHome || (() => setActiveTab('triage'))}
              title="Return to Home Overview & Guide"
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                isHome
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-md font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <Home className="w-4 h-4" />
              <span>Home</span>
            </button>

            <button
              onClick={onGoToTriage || (() => setActiveTab('triage'))}
              title="Point-of-Care Diagnostic Scanner"
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'triage' && !isHome
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-md font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Point-of-Care Triage</span>
            </button>

            <button
              onClick={() => setActiveTab('xai')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'xai'
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-md font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <Brain className="w-4 h-4" />
              <span>Grad-CAM & ABCD</span>
            </button>

            <button
              onClick={() => setActiveTab('research')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                activeTab === 'research'
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-md font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <ShieldCheck className="w-4 h-4" />
              <span>Model & SDG Metrics</span>
            </button>
          </nav>

          {/* Actions & Report Trigger */}
          <div className="flex items-center space-x-2 sm:space-x-3 flex-shrink-0">
            {/* Backend Status Badge */}
            {backendHealth && (
              <div
                className={`hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-semibold border whitespace-nowrap ${
                  backendHealth.online && backendHealth.modelLoaded
                    ? 'bg-emerald-950/40 text-derma-teal border-emerald-500/40 shadow-glow-cyan'
                    : backendHealth.online
                      ? 'bg-amber-950/40 text-amber-300 border-amber-500/40'
                      : 'bg-derma-surface text-slate-400 border-derma-border'
                }`}
                title={
                  backendHealth.online && backendHealth.modelLoaded
                    ? `Connected to PyTorch inference server (${backendHealth.device})`
                    : backendHealth.online
                      ? 'Backend connected (awaiting model weights in models/)'
                      : 'Backend offline - using client-side diagnostic simulation'
                }
              >
                <span
                  className={`w-2 h-2 rounded-full flex-shrink-0 ${
                    backendHealth.online && backendHealth.modelLoaded
                      ? 'bg-derma-teal animate-pulse'
                      : backendHealth.online
                        ? 'bg-amber-400'
                        : 'bg-slate-500'
                  }`}
                />
                <span>
                  {backendHealth.online && backendHealth.modelLoaded
                    ? `AI Engine Live (${backendHealth.device})`
                    : backendHealth.online
                      ? 'Backend Ready'
                      : 'Demo Mode (Client AI)'}
                </span>
              </div>
            )}

            <button
              onClick={onOpenPipeline}
              className="hidden lg:flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-derma-card hover:bg-derma-surface border border-derma-border rounded-lg transition-all whitespace-nowrap"
              title="View Hybrid DenseNet121 + Random Forest Architecture & Formulas"
            >
              <Cpu className="w-3.5 h-3.5 text-derma-accent" />
              <span>ML Architecture</span>
            </button>

            <button
              onClick={onOpenReport}
              className="flex items-center space-x-1.5 sm:space-x-2 px-3 sm:px-4 py-2 text-xs font-bold text-slate-900 bg-derma-accent hover:bg-derma-accentHover rounded-xl shadow-glow-cyan transition-all transform active:scale-95 whitespace-nowrap"
            >
              <FileText className="w-3.5 h-3.5 sm:w-4 sm:h-4 flex-shrink-0" />
              <span>
                <span className="hidden sm:inline">Generate </span>EHR Report
              </span>
            </button>
          </div>
        </div>

        {/* Mobile & Tablet Navigation Bar (< xl) */}
        <div className="xl:hidden flex items-center justify-between gap-1.5 pb-2.5 pt-1 border-t border-derma-border/40 overflow-x-auto no-scrollbar">
          <div className="flex items-center space-x-1 bg-derma-card/80 p-1 rounded-xl border border-derma-border flex-1 min-w-0">
            <button
              onClick={onGoHome || (() => setActiveTab('triage'))}
              className={`flex-1 flex items-center justify-center space-x-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all whitespace-nowrap ${
                isHome
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-sm font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <Home className="w-3.5 h-3.5 flex-shrink-0" />
              <span>Home</span>
            </button>

            <button
              onClick={onGoToTriage || (() => setActiveTab('triage'))}
              className={`flex-1 flex items-center justify-center space-x-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all whitespace-nowrap ${
                activeTab === 'triage' && !isHome
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-sm font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <Activity className="w-3.5 h-3.5 flex-shrink-0" />
              <span>Triage</span>
            </button>

            <button
              onClick={() => setActiveTab('xai')}
              className={`flex-1 flex items-center justify-center space-x-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold transition-all whitespace-nowrap ${
                activeTab === 'xai'
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-sm font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <Brain className="w-3.5 h-3.5 flex-shrink-0" />
              <span>Grad-CAM & ABCD</span>
            </button>

            <button
              onClick={() => setActiveTab('research')}
              className={`flex-1 flex items-center justify-center space-x-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold transition-all whitespace-nowrap ${
                activeTab === 'research'
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-sm font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span>Metrics & SDG</span>
            </button>
          </div>

          <button
            onClick={onOpenPipeline}
            className="lg:hidden flex items-center space-x-1 px-2.5 py-2 text-[11px] font-semibold text-derma-accent bg-derma-card hover:bg-derma-surface border border-derma-border rounded-xl transition-all whitespace-nowrap flex-shrink-0"
            title="View Hybrid DenseNet121 Architecture"
          >
            <Cpu className="w-3.5 h-3.5 flex-shrink-0" />
            <span className="hidden sm:inline">ML Arch</span>
          </button>
        </div>
      </div>
    </header>
  );
}
