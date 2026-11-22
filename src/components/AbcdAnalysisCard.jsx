import React from 'react';
import { Compass, Palette } from 'lucide-react';

const RISK_BAR = 'bg-gradient-to-r from-derma-teal via-derma-yellow to-derma-rose';

function AbcdTile({
  letter,
  letterClass,
  title,
  value,
  unit,
  valueClass,
  status,
  statusClass,
  note,
  percent,
  barClass,
}) {
  return (
    <div className="flex flex-col p-4 rounded-xl bg-derma-bg/90 border border-derma-border hover:border-derma-accent/60 transition-colors">
      <div className="flex items-center gap-2">
        <span
          className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs shrink-0 border ${letterClass}`}
        >
          {letter}
        </span>
        <span className="font-semibold text-slate-50 text-sm">{title}</span>
      </div>

      <div className="mt-3 flex items-baseline gap-1.5">
        <span className={`text-2xl font-semibold font-mono ${valueClass}`}>{value}</span>
        <span className="text-sm text-slate-500 font-mono">{unit}</span>
      </div>
      <p className={`mt-0.5 text-sm font-semibold ${statusClass}`}>{status}</p>

      <p className="mt-3 text-sm text-slate-400 leading-snug flex-1">{note}</p>

      <div className="mt-4 w-full bg-derma-card h-2 rounded-full overflow-hidden border border-derma-border/40">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barClass}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

const pctWidth = (value, max) => Math.min(100, Math.max(5, (value / max) * 100));

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
    <div className="glass-panel p-6 rounded-xl flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-derma-border/70">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-xl bg-derma-teal/20 text-derma-teal border border-derma-teal/30">
              <Compass className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-semibold text-slate-50">
              Dermatological ABCD Rule Feature Analysis
            </h2>
          </div>
          <p className="text-sm text-derma-muted mt-1">
            Quantitative Stolz algorithm scoring: Asymmetry, Border, Color variegation & Diameter
          </p>
        </div>

        {/* TDS Badge */}
        <div
          className={`px-4 py-2 rounded-xl border text-center shrink-0 ${
            isHighRiskTds
              ? 'bg-rose-500/15 border-rose-500/40 text-derma-rose'
              : isBorderlineTds
                ? 'bg-amber-500/15 border-amber-500/40 text-derma-amber'
                : 'bg-emerald-500/15 border-emerald-500/40 text-derma-teal'
          }`}
        >
          <span className="text-xs font-semibold block">Total TDS Score</span>
          <span className="text-xl font-semibold font-mono">{tds}</span>
          <span className="text-xs block font-semibold">
            {isHighRiskTds
              ? 'Malignant Threshold Exceeded (>5.45)'
              : isBorderlineTds
                ? 'Suspicious / Borderline'
                : 'Benign Range (<4.75)'}
          </span>
        </div>
      </div>

      {/* ABCD tiles: auto-fit gives 2 across in a half-width card and 4 across when full width */}
      <div className="grid grid-cols-[repeat(auto-fit,minmax(11rem,1fr))] auto-rows-fr gap-4 flex-1">
        <AbcdTile
          letter="A"
          letterClass="bg-derma-blue/30 text-derma-accent border-derma-blue/40"
          title="Asymmetry"
          value={asymVal.toFixed(1)}
          unit="/ 2.0"
          valueClass="text-derma-accent"
          status={asymVal >= 1.5 ? 'Marked' : asymVal >= 0.8 ? 'Moderate' : 'Minimal'}
          statusClass="text-slate-300"
          note="Evaluated across 2 orthogonal axes (0 = symmetric, 2 = bilateral asymmetry)."
          percent={pctWidth(asymVal, 2)}
          barClass={RISK_BAR}
        />
        <AbcdTile
          letter="B"
          letterClass="bg-purple-500/30 text-purple-300 border-purple-500/40"
          title="Border Cutoff"
          value={borderVal.toFixed(1)}
          unit="/ 8.0"
          valueClass="text-purple-300"
          status={borderVal >= 5.0 ? 'Irregular' : borderVal >= 2.5 ? 'Moderate' : 'Sharp'}
          statusClass="text-slate-300"
          note="Abrupt margin cutoff and jagged indentations across 8 peripheral sectors."
          percent={pctWidth(borderVal, 8)}
          barClass={RISK_BAR}
        />
        <AbcdTile
          letter="C"
          letterClass="bg-amber-500/30 text-derma-amber border-amber-500/40"
          title="Color Count"
          value={colorVal}
          unit="/ 6 shades"
          valueClass="text-derma-amber"
          status={colorVal >= 4 ? 'Variegated' : colorVal >= 2 ? 'Dual-tone' : 'Uniform'}
          statusClass="text-slate-300"
          note="Number of distinct pigment shades detected across the lesion core."
          percent={pctWidth(colorVal, 6)}
          barClass={RISK_BAR}
        />
        <AbcdTile
          letter="D"
          letterClass="bg-emerald-500/30 text-derma-teal border-emerald-500/40"
          title="Diameter"
          value={diamVal.toFixed(1)}
          unit="mm"
          valueClass="text-derma-teal"
          status={diamVal > 6 ? 'Over 6 mm, warning' : 'Under 6 mm, favorable'}
          statusClass={diamVal > 6 ? 'text-derma-rose' : 'text-derma-teal'}
          note={`Maximum horizontal axis span (${diamVal > 6 ? 'exceeds 6 mm threshold' : 'favorable span'}).`}
          percent={pctWidth(diamVal, 12)}
          barClass={diamVal > 6 ? 'bg-derma-rose' : 'bg-derma-teal'}
        />
      </div>

      {/* Detected Colors Tags */}
      <div className="p-4 rounded-xl bg-derma-card/80 border border-derma-border text-sm flex flex-wrap items-center gap-2">
        <span className="font-semibold text-slate-300 flex items-center gap-1.5 mr-2">
          <Palette className="w-4 h-4 text-derma-accent" />
          <span>Pigment Spectrum Identified:</span>
        </span>
        {colorsDetected &&
          colorsDetected.map((col) => (
            <span
              key={col}
              className="px-2.5 py-1 rounded-lg bg-derma-surface text-slate-200 border border-derma-border font-medium text-xs"
            >
              {col}
            </span>
          ))}
      </div>
    </div>
  );
}
