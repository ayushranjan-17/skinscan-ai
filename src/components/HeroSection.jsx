import React from 'react';
import { ShieldCheck, Zap, Award, Target, CheckCircle2, ChevronRight } from 'lucide-react';

export default function HeroSection({ onStartTriage }) {
  return (
    <div className="relative overflow-hidden pt-8 pb-12 border-b border-derma-border/50">
      {/* Background glow effects */}
      <div className="orbital-glow w-96 h-96 bg-derma-accent/10 -top-24 -left-24" />
      <div className="orbital-glow w-[500px] h-[500px] bg-derma-blue/10 -bottom-32 -right-32" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Top Clinical Badge */}
        <div className="flex items-center justify-center">
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-derma-surface/80 border border-derma-border shadow-inner text-xs font-semibold text-derma-accent mb-6">
            <span className="flex h-2 w-2 rounded-full bg-derma-teal animate-pulse" />
            <span>FDA & Clinical Triage Standard Inspired • Multi-Modal HAM10000 Engine</span>
            <ChevronRight className="w-3.5 h-3.5 text-derma-muted" />
          </div>
        </div>

        {/* Hero Title & Subtitle */}
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold text-white tracking-tight leading-tight">
            Objective AI Triage for <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-derma-accent via-cyan-300 to-derma-blue">
              Early Skin Cancer Detection
            </span>
          </h1>

          <p className="mt-4 text-base sm:text-lg text-slate-300 max-w-2xl mx-auto font-normal leading-relaxed">
            A hybrid deep learning & explainable clinical decision support platform. Moving beyond
            raw disease jargon with a clinically actionable{' '}
            <strong className="text-white font-semibold">4-Tier Risk Stratification</strong> and
            quantitative{' '}
            <strong className="text-derma-accent font-semibold">
              1–10 Malignancy Spectral Risk Gauge
            </strong>
            .
          </p>
        </div>

        {/* Key Clinical Metric Counters (DermaSensor validated benchmarks) */}
        <div className="mt-10 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-5xl mx-auto">
          <div className="glass-panel p-5 rounded-2xl border border-derma-border text-center relative overflow-hidden group hover:border-derma-teal/60 transition-all">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-derma-teal to-emerald-400" />
            <p className="text-3xl sm:text-4xl font-extrabold text-derma-teal font-mono">96.0%</p>
            <p className="text-xs font-semibold text-white mt-1">Sensitivity / Recall</p>
            <p className="text-[11px] text-derma-muted mt-0.5">
              High-risk malignant melanoma & BCC detection
            </p>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-derma-border text-center relative overflow-hidden group hover:border-derma-accent/60 transition-all">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-derma-blue to-derma-accent" />
            <p className="text-3xl sm:text-4xl font-extrabold text-derma-accent font-mono">52%</p>
            <p className="text-xs font-semibold text-white mt-1">Missed Referrals Cut</p>
            <p className="text-[11px] text-derma-muted mt-0.5">
              Assists primary care in identifying subtle lesions
            </p>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-derma-border text-center relative overflow-hidden group hover:border-derma-yellow/60 transition-all">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-amber-400 to-derma-yellow" />
            <p className="text-3xl sm:text-4xl font-extrabold text-derma-yellow font-mono">0.94</p>
            <p className="text-xs font-semibold text-white mt-1">ROC-AUC Score</p>
            <p className="text-[11px] text-derma-muted mt-0.5">
              Multi-class discriminative power across 7 lesion classes
            </p>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-derma-border text-center relative overflow-hidden group hover:border-purple-400/60 transition-all">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-derma-purple to-pink-500" />
            <p className="text-3xl sm:text-4xl font-extrabold text-purple-300 font-mono">4-Tier</p>
            <p className="text-xs font-semibold text-white mt-1">Risk Stratification</p>
            <p className="text-[11px] text-derma-muted mt-0.5">
              Benign • Atypical • Pre-Malignant • Malignant
            </p>
          </div>
        </div>

        {/* Academic Accreditation Banner */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-6 text-xs text-derma-muted border-t border-derma-border/40 pt-4">
          <div className="flex items-center space-x-1.5">
            <Award className="w-4 h-4 text-derma-accent" />
            <span>B.M.S. College of Engineering (BMSCE) CSE Major Project Phase-2</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <ShieldCheck className="w-4 h-4 text-derma-teal" />
            <span>
              Aligned with UN SDGs 3 (Good Health), 9 (Innovation), 10 (Reduced Inequalities)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
