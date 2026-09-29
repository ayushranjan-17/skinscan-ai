import React from 'react';
import { ShieldCheck, Activity, Brain, FileText, Cpu, Sparkles } from 'lucide-react';

export default function Navbar({
  activeTab,
  setActiveTab,
  onOpenReport,
  onOpenPipeline,
  backendHealth,
}) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-derma-border/80 bg-derma-bg/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo & Clinical Brand */}
          <div
            className="flex items-center space-x-3 cursor-pointer"
            onClick={() => setActiveTab('triage')}
          >
            <div className="relative flex items-center justify-center w-11 h-11 rounded-xl bg-gradient-to-tr from-derma-blue to-derma-accent p-0.5 shadow-glow-cyan">
              <div className="w-full h-full bg-derma-card rounded-[10px] flex items-center justify-center">
                <Activity className="w-6 h-6 text-derma-accent" />
              </div>
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-derma-teal rounded-full animate-ping" />
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-derma-teal rounded-full" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-white font-sans">
                  SkinScan<span className="text-derma-accent font-extrabold">.AI</span>
                </span>
                <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-derma-surface text-derma-accent rounded border border-derma-border">
                  DermaSensor UI
                </span>
              </div>
              <p className="text-[11px] text-derma-muted font-medium">
                4-Stage Clinical Risk Stratification & Triage
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1 bg-derma-card/80 p-1.5 rounded-xl border border-derma-border">
            <button
              onClick={() => setActiveTab('triage')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'triage'
                  ? 'bg-gradient-to-r from-derma-blue to-derma-accent text-slate-950 shadow-md font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-derma-surface/60'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Point-of-Care Triage</span>
            </button>

            <button
              onClick={() => setActiveTab('xai')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
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
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
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
          <div className="flex items-center space-x-3">
            {/* Backend Status Badge */}
            {backendHealth && (
              <div
                className={`hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-semibold border ${
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
                  className={`w-2 h-2 rounded-full ${
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
              className="hidden lg:flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-derma-card hover:bg-derma-surface border border-derma-border rounded-lg transition-all"
              title="View Hybrid DenseNet121 + Random Forest Architecture & Formulas"
            >
              <Cpu className="w-3.5 h-3.5 text-derma-accent" />
              <span>ML Architecture</span>
            </button>

            <button
              onClick={onOpenReport}
              className="flex items-center space-x-2 px-4 py-2 text-xs font-bold text-slate-900 bg-derma-accent hover:bg-derma-accentHover rounded-xl shadow-glow-cyan transition-all transform active:scale-95"
            >
              <FileText className="w-4 h-4" />
              <span>Generate EHR Report</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
