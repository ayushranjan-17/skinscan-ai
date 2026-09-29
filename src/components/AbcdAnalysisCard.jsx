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

  // Stolz Total Dermoscopy Score (TDS) Interpretation
  const tds =
    totalDermoscopyScore ||
    (asymmetry * 1.3 + border * 0.1 + color * 0.5 + Math.min(5, diameter / 2) * 0.5).toFixed(2);
  const isHighRiskTds = tds > 5.45;
  const isBorderlineTds = tds >= 4.75 && tds <= 5.45;

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
        <div className="p-4 rounded-2xl bg-derma-bg/70 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-derma-muted font-bold uppercase tracking-wider mb-2">
              <span className="flex items-center space-x-1.5 text-white">
                <span className="w-5 h-5 rounded-lg bg-derma-blue/30 text-derma-accent flex items-center justify-center font-bold">
                  A
                </span>
                <span>Asymmetry</span>
              </span>
              <span className="font-mono text-derma-accent">{asymmetry.toFixed(1)} / 2.0</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Evaluated across 2 orthogonal axes (0 = symmetric, 2 = bilateral asymmetry).
            </p>
          </div>
          <div className="mt-4 w-full bg-derma-card h-2 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-derma-teal via-derma-yellow to-derma-rose h-full rounded-full"
              style={{ width: `${(asymmetry / 2) * 100}%` }}
            />
          </div>
        </div>

        {/* B - Border */}
        <div className="p-4 rounded-2xl bg-derma-bg/70 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-derma-muted font-bold uppercase tracking-wider mb-2">
              <span className="flex items-center space-x-1.5 text-white">
                <span className="w-5 h-5 rounded-lg bg-purple-500/30 text-purple-300 flex items-center justify-center font-bold">
                  B
                </span>
                <span>Border</span>
              </span>
              <span className="font-mono text-purple-300">{border.toFixed(1)} / 8.0</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Abrupt margin cutoff and jagged indentations across 8 peripheral sectors.
            </p>
          </div>
          <div className="mt-4 w-full bg-derma-card h-2 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-derma-teal via-derma-yellow to-derma-rose h-full rounded-full"
              style={{ width: `${(border / 8) * 100}%` }}
            />
          </div>
        </div>

        {/* C - Color */}
        <div className="p-4 rounded-2xl bg-derma-bg/70 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-derma-muted font-bold uppercase tracking-wider mb-2">
              <span className="flex items-center space-x-1.5 text-white">
                <span className="w-5 h-5 rounded-lg bg-amber-500/30 text-derma-amber flex items-center justify-center font-bold">
                  C
                </span>
                <span>Color Count</span>
              </span>
              <span className="font-mono text-derma-amber">{color} / 6</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Number of distinct pigment colors detected in the lesion core.
            </p>
          </div>
          <div className="mt-4 w-full bg-derma-card h-2 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-derma-teal via-derma-yellow to-derma-rose h-full rounded-full"
              style={{ width: `${(color / 6) * 100}%` }}
            />
          </div>
        </div>

        {/* D - Diameter */}
        <div className="p-4 rounded-2xl bg-derma-bg/70 border border-derma-border hover:border-derma-accent/60 transition-all flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-derma-muted font-bold uppercase tracking-wider mb-2">
              <span className="flex items-center space-x-1.5 text-white">
                <span className="w-5 h-5 rounded-lg bg-emerald-500/30 text-derma-teal flex items-center justify-center font-bold">
                  D
                </span>
                <span>Diameter</span>
              </span>
              <span className="font-mono text-derma-teal">{diameter.toFixed(1)} mm</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Maximum horizontal axis span. (
              {diameter > 6 ? '>6mm Clinical Warning' : '<6mm Favorable'})
            </p>
          </div>
          <div className="mt-4 w-full bg-derma-card h-2 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${diameter > 6 ? 'bg-derma-rose' : 'bg-derma-teal'}`}
              style={{ width: `${Math.min(100, (diameter / 12) * 100)}%` }}
            />
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
