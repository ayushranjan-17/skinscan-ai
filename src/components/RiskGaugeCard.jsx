import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  AlertOctagon,
  HelpCircle,
  ShieldAlert,
  ArrowUpRight,
  Activity,
} from 'lucide-react';
import { DISEASE_INFO } from '../data/benchmarks';

export default function RiskGaugeCard({ result, onOpenReport }) {
  if (!result) return null;

  const score = result.spectralScore || 1.0;
  const normalizedScore = Math.min(10, Math.max(1, score));
  // Needle rotation calculation for 180-degree semi-circle gauge (from -90deg to +90deg)
  // Score 1 -> -90 deg, Score 10 -> +90 deg
  const needleAngle = -90 + ((normalizedScore - 1) / 9) * 180;

  const getStageColorClass = (stageNum) => {
    switch (stageNum) {
      case 4:
        return 'text-derma-rose bg-rose-500/10 border-rose-500/30';
      case 3:
        return 'text-derma-amber bg-amber-500/10 border-amber-500/30';
      case 2:
        return 'text-derma-yellow bg-yellow-500/10 border-yellow-500/30';
      default:
        return 'text-derma-teal bg-emerald-500/10 border-emerald-500/30';
    }
  };

  const getUrgencyBadge = (urgency) => {
    switch (urgency) {
      case 'Critical':
        return (
          <span className="px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-derma-rose border border-rose-500/40 animate-pulse flex items-center space-x-1.5">
            <AlertOctagon className="w-3.5 h-3.5" />
            <span>Urgent Biopsy Required</span>
          </span>
        );
      case 'High':
        return (
          <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-derma-amber border border-amber-500/40 flex items-center space-x-1.5">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Specialist Review</span>
          </span>
        );
      case 'Moderate':
        return (
          <span className="px-3 py-1 rounded-full text-xs font-bold bg-yellow-500/20 text-derma-yellow border border-yellow-500/40 flex items-center space-x-1.5">
            <Activity className="w-3.5 h-3.5" />
            <span>Observation</span>
          </span>
        );
      default:
        return (
          <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-derma-teal border border-emerald-500/40 flex items-center space-x-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Routine Monitoring</span>
          </span>
        );
    }
  };

  const disease = DISEASE_INFO[result.code] || DISEASE_INFO['mel'];

  return (
    <div className="glass-panel rounded-3xl p-6 border border-derma-border shadow-2xl relative overflow-hidden">
      {/* Background ambient glow based on stage */}
      <div
        className={`orbital-glow w-72 h-72 ${
          result.riskStageNum >= 3 ? 'bg-derma-rose/15' : 'bg-derma-teal/15'
        } -top-16 -right-16`}
      />

      {/* Card Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-derma-border/70 relative z-10">
        <div>
          <span className="text-[11px] font-bold uppercase tracking-widest text-derma-muted">
            Point-of-Care Objective Assessment
          </span>
          <h2 className="text-xl sm:text-2xl font-black text-white mt-0.5">{result.name}</h2>
        </div>
        <div>{getUrgencyBadge(result.urgencyLevel)}</div>
      </div>

      {/* Grid: DermaSensor Gauge + 4-Tier Risk Matrix */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-12 gap-6 relative z-10">
        {/* Left: DermaSensor 1-10 Dial Gauge (5 cols) */}
        <div className="md:col-span-5 bg-derma-bg/70 p-5 rounded-2xl border border-derma-border/90 flex flex-col items-center justify-center text-center">
          <span className="text-xs font-extrabold uppercase tracking-wider text-slate-300 mb-2">
            Malignancy Spectral Score
          </span>

          {/* SVG Gauge Dial */}
          <div className="relative w-56 h-32 flex items-center justify-center overflow-hidden">
            <svg viewBox="0 0 200 115" className="w-full h-full">
              {/* Semi-circular gradient path */}
              <defs>
                <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#00D084" /> {/* Green: 1-3 */}
                  <stop offset="35%" stopColor="#FFDC4B" /> {/* Yellow: 4-5 */}
                  <stop offset="65%" stopColor="#F59E0B" /> {/* Amber: 6-7 */}
                  <stop offset="100%" stopColor="#EF4444" /> {/* Red: 8-10 */}
                </linearGradient>
              </defs>

              {/* Background Arc */}
              <path
                d="M 20 100 A 80 80 0 0 1 180 100"
                fill="none"
                stroke="#17223F"
                strokeWidth="18"
                strokeLinecap="round"
              />

              {/* Colored Gradient Track */}
              <path
                d="M 20 100 A 80 80 0 0 1 180 100"
                fill="none"
                stroke="url(#gaugeGrad)"
                strokeWidth="16"
                strokeLinecap="round"
                strokeDasharray="251.2"
                strokeDashoffset="0"
              />

              {/* Needle */}
              <g transform={`translate(100, 100) rotate(${needleAngle})`}>
                <polygon points="-4,0 4,0 0,-78" fill="#FFFFFF" />
                <circle cx="0" cy="0" r="7" fill="#00F0FF" />
                <circle cx="0" cy="0" r="3" fill="#070B14" />
              </g>

              {/* Ticks */}
              <text
                x="20"
                y="112"
                fill="#8B9BB4"
                fontSize="10"
                fontWeight="bold"
                textAnchor="middle"
              >
                1.0
              </text>
              <text
                x="100"
                y="32"
                fill="#8B9BB4"
                fontSize="10"
                fontWeight="bold"
                textAnchor="middle"
              >
                5.5
              </text>
              <text
                x="180"
                y="112"
                fill="#8B9BB4"
                fontSize="10"
                fontWeight="bold"
                textAnchor="middle"
              >
                10.0
              </text>
            </svg>
          </div>

          {/* Large Digital Score */}
          <div className="mt-1 flex items-baseline space-x-1.5">
            <span className="text-4xl font-black text-white font-mono tracking-tight">
              {score.toFixed(1)}
            </span>
            <span className="text-xs font-bold text-derma-muted font-mono">/ 10.0</span>
          </div>

          {/* DermaSensor Binary Clinical Result */}
          <div className="mt-3 w-full py-2 px-3 rounded-xl bg-derma-card border border-derma-border">
            <p className="text-[11px] text-derma-muted uppercase font-semibold">
              DermaSensor Assessment
            </p>
            <p
              className={`text-sm font-extrabold ${
                score >= 5.5 ? 'text-derma-rose' : 'text-derma-teal'
              }`}
            >
              {score >= 5.5 ? '⚡ INVESTIGATE FURTHER' : '✓ ROUTINE MONITOR'}
            </p>
          </div>
        </div>

        {/* Right: 4-Tier Clinical Risk Stratification (7 cols) */}
        <div className="md:col-span-7 flex flex-col justify-between space-y-4">
          {/* Active Risk Stage Banner */}
          <div className={`p-4 rounded-2xl border ${getStageColorClass(result.riskStageNum)}`}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider">Clinical Risk Tier</span>
              <span className="text-xs font-mono font-bold">
                Classification Confidence: {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <h3 className="text-xl font-extrabold mt-1">{result.riskStage}</h3>
            <p className="text-xs text-slate-300 mt-1 font-medium">{result.actionNeeded}</p>
          </div>

          {/* 4-Stage Progressive Matrix Badges */}
          <div className="grid grid-cols-4 gap-2">
            {[
              { num: 1, label: 'Benign', sub: 'NV/VASC/DF', color: 'teal' },
              { num: 2, label: 'Atypical', sub: 'BKL (Watch)', color: 'yellow' },
              { num: 3, label: 'Pre-Mal', sub: 'AKIEC', color: 'amber' },
              { num: 4, label: 'Malignant', sub: 'MEL / BCC', color: 'rose' },
            ].map((stage) => {
              const isActive = result.riskStageNum === stage.num;
              return (
                <div
                  key={stage.num}
                  className={`p-2 rounded-xl border text-center transition-all ${
                    isActive
                      ? `border-derma-accent bg-derma-surface font-bold text-white shadow-glow-cyan scale-[1.03]`
                      : `border-derma-border/60 bg-derma-card/40 text-slate-500 opacity-60`
                  }`}
                >
                  <span className="text-[10px] font-mono block">Stage {stage.num}</span>
                  <span className="text-xs font-bold block truncate">{stage.label}</span>
                  <span className="text-[9px] block text-derma-muted truncate">{stage.sub}</span>
                </div>
              );
            })}
          </div>

          {/* Clinical Guidance & Suggested ICD-10 */}
          <div className="p-3.5 rounded-xl bg-derma-card/90 border border-derma-border text-xs space-y-1.5">
            <div className="flex items-center justify-between text-derma-muted font-medium">
              <span>ICD-10 Diagnostic Category:</span>
              <span className="font-mono text-slate-300 font-semibold">{result.icd10}</span>
            </div>
            <p className="text-slate-300 text-[11px]">
              <strong className="text-derma-accent">Physician Note: </strong>
              {result.clinicalNotes}
            </p>
          </div>

          {/* NEW: SPECIALIST DOCTOR & CLINICAL ROUTING */}
          {result.referral && (
            <div
              className={`p-4 rounded-2xl border ${
                result.referral.isCancer
                  ? 'bg-rose-950/30 border-rose-500/40 text-rose-100'
                  : 'bg-emerald-950/30 border-derma-teal/40 text-emerald-100'
              }`}
            >
              <div className="flex items-center justify-between pb-2 border-b border-white/10">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-1.5">
                  <span
                    className={`w-2 h-2 rounded-full ${result.referral.isCancer ? 'bg-derma-rose animate-ping' : 'bg-derma-teal'}`}
                  />
                  <span>Recommended Attending Doctor & Routing</span>
                </span>
                <span
                  className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                    result.referral.isCancer
                      ? 'bg-rose-500/30 text-rose-200'
                      : 'bg-emerald-500/30 text-emerald-200'
                  }`}
                >
                  {result.referral.referralTimeline}
                </span>
              </div>

              <div className="mt-2.5 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div>
                  <span className="text-[10px] text-slate-400 block font-medium">
                    Assigned Specialist:
                  </span>
                  <p className="font-extrabold text-white text-sm mt-0.5">
                    {result.referral.doctorName}
                  </p>
                  <p className="text-[11px] text-derma-accent font-semibold">
                    {result.referral.doctorTitle}
                  </p>
                  <p className="text-[10px] text-slate-400 mt-1 font-mono">
                    {result.referral.facility}
                  </p>
                </div>

                <div className="bg-derma-bg/60 p-2.5 rounded-xl border border-derma-border/70 space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block">
                    Pathology Assessment:
                  </span>
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                      result.referral.isCancer
                        ? 'bg-rose-500/20 text-derma-rose border border-rose-500/30'
                        : 'bg-emerald-500/20 text-derma-teal border border-emerald-500/30'
                    }`}
                  >
                    {result.referral.triageClassification}
                  </span>
                  <p className="text-[10px] text-slate-300 mt-1 leading-snug">
                    <strong>Action: </strong>
                    {result.referral.recommendedProcedure}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Class Probabilities Bar Graph */}
      <div className="mt-6 pt-5 border-t border-derma-border/60">
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3">
          7-Class Multi-Head Softmax Probability Distribution
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.entries(result.probabilities || {}).map(([key, prob]) => {
            const diseaseMeta = DISEASE_INFO[key] || { name: key, type: 'Benign' };
            const pct = (prob * 100).toFixed(1);
            const isHighest = key === result.code;
            return (
              <div
                key={key}
                className={`p-2.5 rounded-xl border ${
                  isHighest
                    ? 'border-derma-accent/80 bg-derma-surface/80'
                    : 'border-derma-border/50 bg-derma-bg/60'
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1">
                  <span
                    className={`font-semibold ${isHighest ? 'text-white font-bold' : 'text-slate-300'}`}
                  >
                    {diseaseMeta.name}
                  </span>
                  <span className="font-mono text-[11px] text-derma-accent font-bold">{pct}%</span>
                </div>
                {/* Progress bar */}
                <div className="w-full h-1.5 bg-derma-card rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      diseaseMeta.type === 'Malignant'
                        ? 'bg-gradient-to-r from-derma-amber to-derma-rose'
                        : diseaseMeta.type === 'Pre-Malignant'
                          ? 'bg-derma-amber'
                          : diseaseMeta.type === 'Atypical'
                            ? 'bg-derma-yellow'
                            : 'bg-derma-teal'
                    }`}
                    style={{ width: `${Math.max(2, prob * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
