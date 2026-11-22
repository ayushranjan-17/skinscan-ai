import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Download,
  Database,
  Layers,
  ArrowRight,
  LogIn,
  RefreshCw,
} from 'lucide-react';
import {
  submitBiopsyFeedback,
  getRetrainQueue,
  exportRetrainQueue,
  getUser,
} from '../utils/api';

const BIOPSY_OPTIONS = [
  { code: 'mel', label: 'Melanoma (MEL) — Malignant' },
  { code: 'bcc', label: 'Basal Cell Carcinoma (BCC) — Malignant' },
  { code: 'akiec', label: 'Actinic Keratosis / Bowen\'s (AKIEC) — Pre-Malignant' },
  { code: 'bkl', label: 'Benign Keratosis (BKL) — Benign' },
  { code: 'nv', label: 'Melanocytic Nevus / Mole (NV) — Benign' },
  { code: 'df', label: 'Dermatofibroma (DF) — Benign' },
  { code: 'vasc', label: 'Vascular Lesion (VASC) — Benign' },
  { code: 'wound', label: 'Normal Wound / Ulcer — Non-Neoplastic' },
];

export default function HitlFeedbackPanel({
  scanResult,
  currentUser,
  onOpenAuthModal,
}) {
  const [selectedLabel, setSelectedLabel] = useState(scanResult?.code || 'mel');
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [queueInfo, setQueueInfo] = useState({ queued_count: 0, model_version: 'v4 (frozen)' });
  const [isExporting, setIsExporting] = useState(false);

  // Load MLOps retrain queue status
  const loadQueueStatus = async () => {
    try {
      const q = await getRetrainQueue();
      setQueueInfo(q);
    } catch (err) {
      console.warn('Failed to load MLOps retrain queue:', err);
    }
  };

  useEffect(() => {
    loadQueueStatus();
  }, [submitSuccess]);

  // Sync selected label when scanResult changes
  useEffect(() => {
    if (scanResult?.code) {
      setSelectedLabel(scanResult.code);
    }
  }, [scanResult]);

  const handleSubmitFeedback = async (e) => {
    e.preventDefault();
    if (!scanResult?.scanId) {
      setErrorMsg('This scan is not saved. Sign in to persist scans and attach histopathology feedback.');
      return;
    }
    setIsSubmitting(true);
    setErrorMsg(null);
    setSubmitSuccess(false);

    try {
      await submitBiopsyFeedback(scanResult.scanId, {
        confirmedBiopsyLabel: selectedLabel,
        clinicianNotes: notes,
      });
      setSubmitSuccess(true);
      await loadQueueStatus();
    } catch (err) {
      setErrorMsg(err.message || 'Failed to submit biopsy verification');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleExportManifest = async () => {
    setIsExporting(true);
    try {
      const manifest = await exportRetrainQueue();
      const blob = new Blob([JSON.stringify(manifest, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `skinscan_retrain_manifest_v5_${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="rounded-3xl bg-derma-card border border-derma-border p-6 sm:p-8 space-y-6 shadow-xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-derma-border">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-2xl bg-derma-surface text-derma-accent border border-derma-border shadow-sm">
            <Layers className="w-6 h-6 text-derma-accent" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight flex items-center space-x-2">
              <span>Clinician Biopsy Ground Truth</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-derma-surface text-derma-teal border border-derma-border">
                HITL MLOps
              </span>
            </h3>
            <p className="text-xs text-derma-muted">
              Record verified histopathology labels into the offline v5 retraining queue
            </p>
          </div>
        </div>

        {/* Offline Queue Badge & Export Button */}
        <div className="flex items-center space-x-2 self-start sm:self-auto">
          <div className="text-right">
            <div className="text-[10px] font-mono text-slate-400">Offline v5 Queue</div>
            <div className="text-xs font-bold text-derma-teal font-mono">
              {queueInfo.queued_count} cases queued
            </div>
          </div>
          <button
            onClick={handleExportManifest}
            disabled={isExporting || queueInfo.queued_count === 0}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-slate-200 bg-derma-surface hover:bg-derma-border border border-derma-border transition-all disabled:opacity-40"
            title="Download anonymized JSON retraining manifest for offline training"
          >
            <Download className="w-3.5 h-3.5 text-derma-accent" />
            <span>Export Manifest</span>
          </button>
        </div>
      </div>

      {/* Guest Mode Notice */}
      {!currentUser && (
        <div className="p-4 rounded-2xl bg-amber-950/20 border border-amber-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center space-x-2 text-amber-300">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>
              You are currently in <strong>Guest Mode (Zero-Storage)</strong>. Sign in to save scans and queue biopsy cases.
            </span>
          </div>
          <button
            onClick={onOpenAuthModal}
            className="flex items-center space-x-1.5 px-3 py-1.5 bg-gradient-to-r from-derma-accent to-derma-blue text-slate-950 font-bold rounded-xl shadow-glow-cyan transition-all"
          >
            <LogIn className="w-3.5 h-3.5" />
            <span>Sign In Clinician</span>
          </button>
        </div>
      )}

      {/* Biopsy Feedback Form */}
      {scanResult?.scanId ? (
        <form onSubmit={handleSubmitFeedback} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Confirmed Histopathology Biopsy Label:
              </label>
              <select
                value={selectedLabel}
                onChange={(e) => setSelectedLabel(e.target.value)}
                className="w-full px-3.5 py-2.5 bg-derma-surface border border-derma-border rounded-xl text-xs text-white focus:outline-none focus:border-derma-accent transition-colors"
              >
                {BIOPSY_OPTIONS.map((opt) => (
                  <option key={opt.code} value={opt.code}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Histopathology & Surgical Notes (Optional):
              </label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g., Clark Level II, Breslow depth 0.4mm, clear margins"
                className="w-full px-3.5 py-2.5 bg-derma-surface border border-derma-border rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-derma-accent transition-colors"
              />
            </div>
          </div>

          {submitSuccess && (
            <div className="flex items-center space-x-2 p-3 text-xs text-emerald-300 bg-emerald-950/40 border border-emerald-500/30 rounded-xl">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <span>Biopsy ground truth attached and safely queued for offline v5 retraining!</span>
            </div>
          )}

          {errorMsg && (
            <div className="flex items-center space-x-2 p-3 text-xs text-rose-300 bg-rose-950/40 border border-rose-500/30 rounded-xl">
              <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="flex items-center space-x-2 px-5 py-2.5 text-xs font-bold text-slate-950 bg-gradient-to-r from-derma-teal to-derma-accent hover:brightness-110 rounded-xl shadow-glow-cyan transition-all disabled:opacity-50"
          >
            <CheckCircle2 className="w-4 h-4" />
            <span>{isSubmitting ? 'Recording Feedback...' : 'Verify & Queue for Offline Retraining'}</span>
          </button>
        </form>
      ) : (
        <div className="p-4 rounded-2xl bg-derma-surface/60 border border-derma-border text-xs text-slate-400 flex items-center justify-between">
          <span>
            Run an authenticated scan to attach biopsy ground truth. Scans evaluated in Guest Mode are stateless.
          </span>
          {currentUser && (
            <span className="font-mono text-derma-teal font-semibold">User: {currentUser.name}</span>
          )}
        </div>
      )}
    </div>
  );
}

