import React, { useState } from 'react';
import { BarChart3, Cpu, Globe, HeartPulse, FlaskConical } from 'lucide-react';
import metrics from '../data/modelMetrics.json';

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const dec = (v) => (v == null ? '—' : v.toFixed(3));

const TABS = [
  { id: 'confusion', label: 'Confusion matrix' },
  { id: 'roc', label: 'ROC-AUC' },
  { id: 'comparison', label: 'Model comparison' },
  { id: 'sdg', label: 'SDG goals' },
];

const ROC_COLORS = ['#E5645A', '#E8A33D', '#EBC95A', '#34C38F', '#4FD1E8', '#A77FDB', '#4A9FE0'];

function RocChart({ classes, active }) {
  const W = 520;
  const H = 400;
  const pad = { l: 52, r: 16, t: 16, b: 48 };
  const x = (v) => pad.l + v * (W - pad.l - pad.r);
  const y = (v) => H - pad.b - v * (H - pad.t - pad.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const on = (code) => active === null || active === code;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="ROC curves, one line per class">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#1F2E54" />
          <line x1={x(t)} x2={x(t)} y1={pad.t} y2={H - pad.b} stroke="#1F2E54" />
          <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" fontSize="12" fill="#8B9BB4">
            {t}
          </text>
          <text x={x(t)} y={H - pad.b + 18} textAnchor="middle" fontSize="12" fill="#8B9BB4">
            {t}
          </text>
        </g>
      ))}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="#4A5568" strokeDasharray="5 5" />
      {classes.map(
        (c, i) =>
          c.roc && (
            <polyline
              key={c.code}
              points={c.roc.map(([f, t]) => `${x(f)},${y(t)}`).join(' ')}
              fill="none"
              stroke={ROC_COLORS[i]}
              strokeWidth={on(c.code) ? 2.5 : 1.5}
              strokeLinejoin="round"
              opacity={on(c.code) ? 1 : 0.2}
            />
          )
      )}
      <text x={(pad.l + W - pad.r) / 2} y={H - 8} textAnchor="middle" fontSize="12" fill="#8B9BB4">
        False positive rate
      </text>
      <text
        transform={`translate(14 ${(pad.t + H - pad.b) / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize="12"
        fill="#8B9BB4"
      >
        True positive rate
      </text>
    </svg>
  );
}

function NotEvaluated({ what }) {
  return (
    <div className="rounded-xl border border-dashed border-derma-borderLight p-8 text-center">
      <FlaskConical className="w-6 h-6 mx-auto text-derma-muted" />
      <p className="mt-3 font-semibold text-slate-50">{what} hasn&apos;t been measured yet</p>
      <p className="mt-1 text-sm text-derma-muted max-w-md mx-auto">
        Run <code className="text-slate-200">ml_pipeline/evaluate_metrics.py</code> on a held-out
        test set. It writes <code className="text-slate-200">src/data/modelMetrics.json</code>, and
        this page fills in from that file.
      </p>
    </div>
  );
}

export default function ResearchMetricsDashboard() {
  const [metricTab, setMetricTab] = useState('confusion');
  const [activeClass, setActiveClass] = useState(null);
  const evaluated = metrics.status === 'evaluated';
  const classes = metrics.classes;
  const cm = metrics.confusion_matrix;
  const hasRoc = classes.some((c) => c.roc);
  const comparisonRows = [
    {
      name: 'SkinScan v4 (DenseNet121 + patient data)',
      accuracy: metrics.overall_accuracy,
      malignant_recall: metrics.malignant_sensitivity,
      auc: metrics.macro_auc,
      source: 'This project',
    },
    ...metrics.comparison,
  ];

  return (
    <div className="glass-panel p-6 sm:p-8 rounded-xl space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-derma-border/70">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-xl bg-derma-surface text-derma-accent border border-derma-border">
              <BarChart3 className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-semibold text-slate-50">How well does it work?</h2>
          </div>
          <p className="text-sm text-derma-muted mt-1">
            {evaluated
              ? `${metrics.dataset} · ${metrics.n_test} test images · ${metrics.model}`
              : `No evaluation results loaded yet for ${metrics.model}.`}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1 bg-derma-bg p-1.5 rounded-xl border border-derma-border text-xs font-semibold">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setMetricTab(t.id)}
              className={`px-3 py-1.5 rounded-lg transition-colors ${
                metricTab === t.id
                  ? 'bg-derma-surface text-derma-accent border border-derma-border'
                  : 'text-slate-400 hover:text-slate-50'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {evaluated && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {[
              ['Accuracy', pct(metrics.overall_accuracy)],
              ['Macro AUC', dec(metrics.macro_auc)],
              ['Melanoma + BCC caught (top prediction)', pct(metrics.malignant_sensitivity)],
              metrics.cancer_alert_sensitivity != null && [
                `Melanoma + BCC flagged (alert at ${pct(metrics.alert_threshold)})`,
                pct(metrics.cancer_alert_sensitivity),
              ],
              metrics.alert_specificity != null && [
                `Benign cleared (alert at ${pct(metrics.alert_threshold)})`,
                pct(metrics.alert_specificity),
              ],
            ]
              .filter(Boolean)
              .map(([label, value]) => (
                <div key={label} className="rounded-xl bg-derma-bg/70 border border-derma-border p-3">
                  <p className="text-xs text-derma-muted">{label}</p>
                  <p className="mt-1 text-xl font-semibold text-slate-50 font-mono">{value}</p>
                </div>
              ))}
          </div>
          <p className="text-sm text-derma-muted">
            &quot;Top prediction&quot; counts a lesion as caught only if the single most likely class
            is melanoma or BCC. The app also raises a biopsy alert when the combined melanoma, BCC
            and actinic keratosis probability reaches the alert level, and that is what a clinician
            sees. The level was chosen on separate validation lesions to catch about 90% of
            melanoma and BCC, then measured here on the test lesions. These figures use model
            probabilities only, not the extra checks in the app.
          </p>
        </div>
      )}

      {metricTab === 'confusion' &&
        (!evaluated || !cm ? (
          <NotEvaluated what="The confusion matrix" />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-7 bg-derma-bg/80 p-5 rounded-xl border border-derma-border overflow-x-auto">
              <div className="flex items-center justify-between mb-3 text-xs">
                <span className="font-semibold text-slate-50">7-class confusion matrix</span>
                <span className="text-derma-muted">Rows: true class · Columns: predicted</span>
              </div>
              <table className="w-full text-center text-xs font-mono border-collapse">
                <thead>
                  <tr className="text-derma-accent">
                    <th className="p-2 text-left text-slate-400 font-sans">True \ Pred</th>
                    {classes.map((c) => (
                      <th key={c.code} className="p-1.5">
                        {c.code}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-derma-border/50">
                  {cm.map((row, r) => (
                    <tr key={classes[r].code}>
                      <td className="p-1.5 text-left font-semibold text-slate-300 font-sans">
                        {classes[r].code}
                      </td>
                      {row.map((val, c) => (
                        <td
                          key={classes[c].code}
                          className={`p-2 rounded font-semibold ${
                            r === c
                              ? 'bg-derma-teal/25 text-emerald-300 border border-derma-teal/40'
                              : val > 5
                                ? 'bg-amber-500/20 text-derma-amber'
                                : 'text-slate-500'
                          }`}
                        >
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="lg:col-span-5 bg-derma-card p-5 rounded-xl border border-derma-border space-y-3">
              <h3 className="text-base font-semibold text-slate-50">Per-class results</h3>
              {classes.map((cls) => (
                <div
                  key={cls.code}
                  className="p-2.5 rounded-xl bg-derma-bg/70 border border-derma-border flex items-center justify-between text-sm"
                >
                  <div>
                    <span className="font-semibold text-slate-50 block">
                      {cls.name} ({cls.code})
                    </span>
                    <span className="text-xs text-derma-muted font-mono">F1 {dec(cls.f1)}</span>
                  </div>
                  <div className="text-right font-mono">
                    <span className="text-derma-teal font-semibold block">
                      Recall {pct(cls.recall)}
                    </span>
                    <span className="text-derma-accent text-xs">Precision {pct(cls.precision)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

      {metricTab === 'roc' &&
        (!evaluated || metrics.macro_auc == null ? (
          <NotEvaluated what="ROC-AUC" />
        ) : (
          <div className="p-6 bg-derma-bg/80 rounded-xl border border-derma-border space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold text-slate-50">
                  ROC curves, each class against the rest
                </h3>
                <p className="text-sm text-derma-muted">
                  The closer a curve hugs the top-left corner, the better. The dashed line is
                  guessing. Hover a class on the right to isolate its curve.
                </p>
              </div>
              <span className="px-3 py-1.5 rounded-xl bg-derma-surface text-derma-accent border border-derma-border font-mono text-xs font-semibold">
                Macro AUC {dec(metrics.macro_auc)}
              </span>
            </div>

            <div className="grid grid-cols-[repeat(auto-fit,minmax(20rem,1fr))] gap-6 items-center">
              {hasRoc && <RocChart classes={classes} active={activeClass} />}
              <ul className="space-y-2" onMouseLeave={() => setActiveClass(null)}>
                {classes.map((c, i) => (
                  <li
                    key={c.code}
                    onMouseEnter={() => setActiveClass(c.code)}
                    className="flex items-center gap-3 rounded-xl bg-derma-card border border-derma-border px-3 py-2 text-sm"
                  >
                    <span
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: ROC_COLORS[i] }}
                    />
                    <span className="text-slate-200 flex-1">{c.name}</span>
                    <span className="font-mono text-slate-50">{dec(c.auc)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}

      {metricTab === 'comparison' &&
        (metrics.comparison.length === 0 ? (
          <NotEvaluated what="The comparison against other models" />
        ) : (
          <div className="bg-derma-bg/80 p-5 rounded-xl border border-derma-border space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-derma-accent">
                  <tr>
                    <th className="p-3">Model</th>
                    <th className="p-3">Accuracy</th>
                    <th className="p-3">Melanoma + BCC caught (top prediction)</th>
                    <th className="p-3">Macro AUC</th>
                    <th className="p-3">How it was built</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-derma-border/70 text-slate-300">
                  {comparisonRows.map((m, i) => (
                    <tr key={m.name} className={i === 0 ? 'bg-derma-card text-slate-50' : ''}>
                      <td className={`p-3 ${i === 0 ? 'font-semibold text-derma-accent' : 'font-medium'}`}>
                        {m.name}
                      </td>
                      <td className="p-3 font-mono">{pct(m.accuracy)}</td>
                      <td className="p-3 font-mono">{pct(m.malignant_recall)}</td>
                      <td className="p-3 font-mono">{dec(m.auc)}</td>
                      <td className="p-3 text-derma-muted">{m.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-derma-muted">
              Every row is scored on the same {metrics.n_test} test images. The baselines are
              trained on the remaining lesions with balanced class weights, which pushes them to
              guess the rare cancer classes more often. That can raise the &quot;caught&quot; column
              while lowering accuracy, so macro AUC is the fairer comparison. This table shows what
              the image and the fine-tuning add; it is not a ranking against published systems.
            </p>
          </div>
        ))}

      {metricTab === 'sdg' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-xl bg-derma-surface border border-emerald-500/30 space-y-2">
            <div className="flex items-center space-x-2 text-derma-teal font-semibold">
              <HeartPulse className="w-5 h-5" />
              <span>SDG 3: Good health</span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              Target 3.4: reduce premature deaths from skin cancer by helping catch it earlier, at
              the point of care.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-derma-surface border border-derma-blue/30 space-y-2">
            <div className="flex items-center space-x-2 text-derma-accent font-semibold">
              <Cpu className="w-5 h-5" />
              <span>SDG 9: Innovation</span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              Combines image and patient data in one model and shows its reasoning with Grad-CAM,
              so clinicians can check it.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-derma-surface border border-derma-purple/30 space-y-2">
            <div className="flex items-center space-x-2 text-purple-300 font-semibold">
              <Globe className="w-5 h-5" />
              <span>SDG 10: Reduced inequalities</span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              Aims to bring dermatology-style triage to clinics that don&apos;t have a specialist
              nearby.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
