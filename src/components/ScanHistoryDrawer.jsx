import React, { useState, useEffect } from 'react';
import {
  X,
  History,
  Trash2,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  FileText,
  User,
  ShieldCheck,
  Sparkles,
  ExternalLink,
} from 'lucide-react';
import { getScans, deleteScan } from '../utils/api';
import { BIOPSY_ALERT_THRESHOLD } from '../data/alertThreshold';

export default function ScanHistoryDrawer({
  isOpen,
  onClose,
  currentUser,
  onSelectScan,
  onOpenFeedback,
}) {
  const [scans, setScans] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [deleteId, setDeleteId] = useState(null);

  const fetchScans = async () => {
    if (!currentUser) return;
    setIsLoading(true);
    try {
      const data = await getScans();
      setScans(data);
    } catch (err) {
      console.warn('Failed to load scan history:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchScans();
    }
  }, [isOpen, currentUser]);

  const handleDelete = async (scanId) => {
    try {
      setDeleteId(scanId);
      await deleteScan(scanId);
      setScans((prev) => prev.filter((s) => s.id !== scanId));
    } catch (err) {
      alert(`Could not delete scan: ${err.message}`);
    } finally {
      setDeleteId(null);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-xl h-full bg-derma-bg border-l border-derma-border shadow-2xl flex flex-col overflow-hidden">
        {/* Drawer Header */}
        <div className="p-6 border-b border-derma-border/80 flex items-center justify-between bg-derma-card/80">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-derma-surface text-derma-accent border border-derma-border shadow-sm">
              <History className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-50 tracking-tight flex items-center space-x-2">
                <span>Persistent Scan History</span>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-derma-surface text-derma-teal border border-derma-border">
                  SQLite
                </span>
              </h2>
              <p className="text-xs text-derma-muted">
                {currentUser?.name} • {scans.length} recorded {scans.length === 1 ? 'scan' : 'scans'}
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-50 hover:bg-derma-surface transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scans Content List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {isLoading ? (
            <div className="py-20 text-center space-y-3">
              <div className="w-8 h-8 mx-auto border-2 border-derma-accent border-t-transparent rounded-full animate-spin" />
              <p className="text-xs text-slate-400">Loading clinical records from SQLite...</p>
            </div>
          ) : scans.length === 0 ? (
            <div className="py-20 text-center space-y-4">
              <div className="w-14 h-14 mx-auto rounded-xl bg-derma-surface border border-derma-border flex items-center justify-center text-slate-500">
                <FileText className="w-7 h-7" />
              </div>
              <div className="max-w-xs mx-auto">
                <h3 className="text-sm font-bold text-slate-50">No scans recorded yet</h3>
                <p className="text-xs text-derma-muted mt-1">
                  Upload and analyze a skin lesion while signed in to automatically store cases here.
                </p>
              </div>
            </div>
          ) : (
            scans.map((scan) => {
              const isMalignant =
                scan.risk_stage?.includes('4') ||
                scan.predicted_class?.toLowerCase().includes('melanoma') ||
                scan.predicted_class?.toLowerCase().includes('carcinoma') ||
                (scan.malignant_probability || 0) >= BIOPSY_ALERT_THRESHOLD;

              return (
                <div
                  key={scan.id}
                  className="p-4 rounded-xl bg-derma-card/90 border border-derma-border hover:border-derma-accent/50 transition-all space-y-3"
                >
                  {/* Top Bar: Disease, Stage & Date */}
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-bold text-slate-50">
                          {scan.predicted_class || 'Lesion Triage'}
                        </span>
                        {scan.icd10_code && (
                          <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-derma-surface text-slate-300 border border-derma-border">
                            {scan.icd10_code}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-3 text-xs text-derma-muted mt-1">
                        <span className="flex items-center space-x-1">
                          <Calendar className="w-3 h-3 text-slate-500" />
                          <span>{scan.created_at ? new Date(scan.created_at).toLocaleDateString() : 'Recent'}</span>
                        </span>
                        <span>•</span>
                        <span>
                          {scan.patient_age ? `${scan.patient_age}y` : 'Age N/A'}, {scan.patient_sex || 'Sex N/A'}
                        </span>
                        <span>•</span>
                        <span className="capitalize">{scan.body_site || 'Site N/A'}</span>
                      </div>
                    </div>

                    <div className="text-right flex-shrink-0">
                      <span
                        className={`inline-block px-2.5 py-1 text-[11px] font-bold rounded-lg ${
                          isMalignant
                            ? 'bg-rose-950/60 text-rose-300 border border-rose-500/40 shadow-[0_0_12px_rgba(244,63,94,0.25)]'
                            : 'bg-emerald-950/60 text-emerald-300 border border-emerald-500/40'
                        }`}
                      >
                        {scan.risk_stage || 'Stage 1'}
                      </span>
                      <div className="text-xs font-mono text-derma-accent font-semibold mt-1">
                        Score: {scan.spectral_score?.toFixed(1) || '0.0'}/10
                      </div>
                    </div>
                  </div>

                  {/* Biopsy Ground Truth Status */}
                  <div className="pt-2 border-t border-derma-border/50 flex items-center justify-between gap-2 text-xs">
                    <div className="flex items-center space-x-2">
                      {scan.queued_for_retrain === 1 ? (
                        <span className="flex items-center space-x-1 text-derma-teal text-xs font-semibold bg-emerald-950/30 px-2 py-0.5 rounded border border-emerald-500/30">
                          <CheckCircle2 className="w-3 h-3 text-derma-teal" />
                          <span>Biopsy: {scan.confirmed_biopsy_label?.toUpperCase()} (Queued v5)</span>
                        </span>
                      ) : (
                        <span className="text-xs text-slate-400">Biopsy Ground Truth: Pending</span>
                      )}
                    </div>

                    <div className="flex items-center space-x-2">
                      {onOpenFeedback && (
                        <button
                          onClick={() => {
                            onOpenFeedback(scan);
                            onClose();
                          }}
                          className="px-2.5 py-1 text-xs font-semibold text-derma-accent hover:text-slate-50 bg-derma-surface hover:bg-derma-accent/20 rounded-lg border border-derma-border transition-all"
                        >
                          Attach Biopsy
                        </button>
                      )}

                      <button
                        onClick={() => handleDelete(scan.id)}
                        disabled={deleteId === scan.id}
                        title="Delete scan record"
                        className="p-1 text-slate-500 hover:text-rose-400 hover:bg-rose-950/40 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

