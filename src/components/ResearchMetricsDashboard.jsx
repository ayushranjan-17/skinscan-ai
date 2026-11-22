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
  const evaluated = metrics.status === 'evaluated';
  const classes = metrics.classes;
  const cm = metrics.confusion_matrix;

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
              metrics.alert_sensitivity != null && [
                `Cancers flagged (alert at ${pct(metrics.alert_threshold)})`,
                pct(metrics.alert_sensitivity),
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
            sees. These figures use model probabilities only, not the extra checks in the app.
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
          <div className="p-6 bg-derma-bg/80 rounded-xl border border-derma-border space-y-4">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-base font-semibold text-slate-50">
                Area under the ROC curve, one class against the rest
              </h3>
              <span className="px-3 py-1.5 rounded-xl bg-derma-surface text-derma-accent border border-derma-border font-mono text-xs font-semibold">
                Macro AUC {dec(metrics.macro_auc)}
              </span>
            </div>
            <ul className="grid sm:grid-cols-2 gap-2">
              {classes.map((c) => (
                <li
                  key={c.code}
                  className="flex items-center justify-between rounded-xl bg-derma-card border border-derma-border px-3 py-2 text-sm"
                >
                  <span className="text-slate-200">{c.name}</span>
                  <span className="font-mono text-derma-accent">{dec(c.auc)}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}

      {metricTab === 'comparison' &&
        (metrics.comparison.length === 0 ? (
          <NotEvaluated what="The comparison against other models" />
        ) : (
          <div className="bg-derma-bg/80 p-5 rounded-xl border border-derma-border overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-derma-accent">
                <tr>
                  <th className="p-3">Model</th>
                  <th className="p-3">Accuracy</th>
                  <th className="p-3">Malignant recall</th>
                  <th className="p-3">AUC</th>
                  <th className="p-3">Source</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-derma-border/70 text-slate-300">
                {metrics.comparison.map((m) => (
                  <tr key={m.name}>
                    <td className="p-3 font-semibold text-slate-50">{m.name}</td>
                    <td className="p-3 font-mono">{pct(m.accuracy)}</td>
                    <td className="p-3 font-mono">{pct(m.malignant_recall)}</td>
                    <td className="p-3 font-mono">{dec(m.auc)}</td>
                    <td className="p-3 text-derma-muted">{m.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
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
