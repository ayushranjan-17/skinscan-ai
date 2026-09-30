import React from 'react';
import {
  Sparkles,
  Check,
  AlertTriangle,
  ShieldCheck,
  Ruler,
  Palette,
  Compass,
  Layers,
} from 'lucide-react';

export default function AbcdAnalysisCard({ result }) {
  if (!result || !result.abcd) return null;

  const { asymmetry, border, color, diameter, colorsDetected, totalDermoscopyScore } = result.abcd;

  const asymVal = Math.min(2.0, Math.max(0.0, Number(asymmetry || 0)));
  const borderVal = Math.min(8.0, Math.max(0.0, Number(border || 0)));
  const colorVal = Math.min(6, Math.max(1, Number(color || 1)));
  const diamVal = Math.max(0.1, Number(diameter || 1.0));

  // Stolz Total Dermoscopy Score (TDS) Interpretation
  const tds =
    totalDermoscopyScore ||
    (asymVal * 1.3 + borderVal * 0.1 + colorVal * 0.5 + Math.min(5, diamVal / 2) * 0.5).toFixed(1);
  const isHighRiskTds = Number(tds) > 5.45;
  const isBorderlineTds = Number(tds) >= 4.75 && Number(tds) <= 5.45;

  return (
    <div className="glass-panel p-6 rounded-3xl border border-derma-border shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-derma-border/70">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-xl bg-derma-teal/20 text-derma-teal border border-derma-teal/30">
              <Compass className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-bold text-white font-sans">
              Dermatological ABCD Rule Feature Analysis
            </h2>
          </div>
          <p className="text-xs text-derma-muted mt-1">
            Quantitative Stolz algorithm scoring: Asymmetry, Border, Color variegation & Diameter
          </p>
        </div>

        {/* TDS Badge */}
        <div
          className={`px-4 py-2 rounded-2xl border text-center ${
            isHighRiskTds
              ? 'bg-rose-500/15 border-rose-500/40 text-derma-rose'
              : isBorderlineTds
                ? 'bg-amber-500/15 border-amber-500/40 text-derma-amber'
                : 'bg-emerald-500/15 border-emerald-500/40 text-derma-teal'
          }`}
        >
          <span className="text-[10px] uppercase font-bold tracking-wider block">
            Total TDS Score
          </span>
          <span className="text-lg font-black font-mono">{tds}</span>
          <span className="text-[10px] block font-semibold">
            {isHighRiskTds
              ? 'Malignant Threshold Exceeded (>5.45)'
              : isBorderlineTds
                ? 'Suspicious / Borderline'
                : 'Benign Range (<4.75)'}
          </span>
        </div>
      </div>

      {/* ABCD 4 Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* A - Asymmetry */}
        <div className="p-4 rounded-2xl bg-derma-bg/90 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between space-y-3">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-lg bg-derma-blue/30 text-derma-accent flex items-center justify-center font-black text-xs shrink-0 border border-derma-blue/40">
                A
              </span>
              <span className="font-extrabold text-white text-xs uppercase tracking-wider">
                Asymmetry
              </span>
            </div>
            <div className="flex items-baseline justify-between pt-0.5">
              <div className="flex items-baseline space-x-1">
                <span className="text-xl font-black font-mono text-derma-accent">
                  {asymVal.toFixed(1)}
                </span>
                <span className="text-xs text-slate-500 font-mono font-medium">/ 2.0</span>
              </div>
              <span className="text-[10px] uppercase font-bold text-slate-400">
                {asymVal >= 1.5 ? 'Marked' : asymVal >= 0.8 ? 'Moderate' : 'Minimal'}
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-derma-card h-2 rounded-full overflow-hidden border border-derma-border/40">
              <div
                className="bg-gradient-to-r from-derma-teal via-derma-yellow to-derma-rose h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(5, (asymVal / 2) * 100))}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400 leading-snug">
              Evaluated across 2 orthogonal axes (0 = symmetric, 2 = bilateral asymmetry).
            </p>
          </div>
        </div>

        {/* B - Border */}
        <div className="p-4 rounded-2xl bg-derma-bg/90 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between space-y-3">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-lg bg-purple-500/30 text-purple-300 flex items-center justify-center font-black text-xs shrink-0 border border-purple-500/40">
                B
              </span>
              <span className="font-extrabold text-white text-xs uppercase tracking-wider">
                Border Cutoff
              </span>
            </div>
            <div className="flex items-baseline justify-between pt-0.5">
              <div className="flex items-baseline space-x-1">
                <span className="text-xl font-black font-mono text-purple-300">
                  {borderVal.toFixed(1)}
                </span>
                <span className="text-xs text-slate-500 font-mono font-medium">/ 8.0</span>
              </div>
              <span className="text-[10px] uppercase font-bold text-slate-400">
                {borderVal >= 5.0 ? 'Irregular' : borderVal >= 2.5 ? 'Moderate' : 'Sharp'}
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-derma-card h-2 rounded-full overflow-hidden border border-derma-border/40">
              <div
                className="bg-gradient-to-r from-derma-teal via-derma-yellow to-derma-rose h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(5, (borderVal / 8) * 100))}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400 leading-snug">
              Abrupt margin cutoff and jagged indentations across 8 peripheral sectors.
            </p>
          </div>
        </div>

        {/* C - Color */}
        <div className="p-4 rounded-2xl bg-derma-bg/90 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between space-y-3">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-lg bg-amber-500/30 text-derma-amber flex items-center justify-center font-black text-xs shrink-0 border border-amber-500/40">
                C
              </span>
              <span className="font-extrabold text-white text-xs uppercase tracking-wider">
                Color Count
              </span>
            </div>
            <div className="flex items-baseline justify-between pt-0.5">
              <div className="flex items-baseline space-x-1">
                <span className="text-xl font-black font-mono text-derma-amber">{colorVal}</span>
                <span className="text-xs text-slate-500 font-mono font-medium">/ 6 shades</span>
              </div>
              <span className="text-[10px] uppercase font-bold text-slate-400">
                {colorVal >= 4 ? 'Variegated' : colorVal >= 2 ? 'Dual-Tone' : 'Uniform'}
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-derma-card h-2 rounded-full overflow-hidden border border-derma-border/40">
              <div
                className="bg-gradient-to-r from-derma-teal via-derma-yellow to-derma-rose h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(5, (colorVal / 6) * 100))}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400 leading-snug">
              Number of distinct pigment shades detected across the lesion core.
            </p>
          </div>
        </div>

        {/* D - Diameter */}
        <div className="p-4 rounded-2xl bg-derma-bg/90 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between space-y-3">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-lg bg-emerald-500/30 text-derma-teal flex items-center justify-center font-black text-xs shrink-0 border border-emerald-500/40">
                D
              </span>
              <span className="font-extrabold text-white text-xs uppercase tracking-wider">
                Diameter
              </span>
            </div>
            <div className="flex items-baseline justify-between pt-0.5">
              <div className="flex items-baseline space-x-1">
                <span className="text-xl font-black font-mono text-derma-teal">
                  {diamVal.toFixed(1)}
                </span>
                <span className="text-xs text-slate-500 font-mono font-medium">mm</span>
              </div>
              <span
                className={`text-[10px] uppercase font-bold ${diamVal > 6 ? 'text-derma-rose font-black' : 'text-derma-teal'}`}
              >
                {diamVal > 6 ? '>6mm Warning' : '<6mm Favorable'}
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-derma-card h-2 rounded-full overflow-hidden border border-derma-border/40">
              <div
                className={`h-full rounded-full transition-all duration-500 ${diamVal > 6 ? 'bg-derma-rose' : 'bg-derma-teal'}`}
                style={{ width: `${Math.min(100, Math.max(5, (diamVal / 12) * 100))}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400 leading-snug">
              Maximum horizontal axis span (
              {diamVal > 6 ? 'Exceeds 6mm threshold' : 'Favorable span'}).
            </p>
          </div>
        </div>
      </div>

      {/* Detected Colors Tags */}
      <div className="p-4 rounded-2xl bg-derma-card/80 border border-derma-border text-xs flex flex-wrap items-center gap-2">
        <span className="font-bold text-slate-300 flex items-center space-x-1.5 mr-2">
          <Palette className="w-4 h-4 text-derma-accent" />
          <span>Pigment Spectrum Identified:</span>
        </span>
        {colorsDetected &&
          colorsDetected.map((col, idx) => (
            <span
              key={idx}
              className="px-2.5 py-1 rounded-lg bg-derma-surface text-slate-200 border border-derma-border font-medium text-[11px]"
            >
              {col}
            </span>
          ))}
      </div>
    </div>
  );
}
