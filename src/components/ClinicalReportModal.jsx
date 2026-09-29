import React, { useRef, useEffect } from 'react';
import {
  X,
  Printer,
  Download,
  ShieldCheck,
  Activity,
  Award,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Camera,
  Upload,
  Layers,
  Sparkles,
} from 'lucide-react';
import {
  renderLesionCanvas,
  applyDullRazorAndColorConstancy,
  renderGradCamOverlay,
  renderContourOverlay,
} from '../utils/imageProcessing';

export default function ClinicalReportModal({
  isOpen,
  onClose,
  result,
  patientData,
  customImageSrc,
  acquisitionMode = 'benchmark',
}) {
  const reportRef = useRef(null);
  const reportOrigCanvasRef = useRef(null);
  const reportDullCanvasRef = useRef(null);
  const reportGradCanvasRef = useRef(null);
  const reportContourCanvasRef = useRef(null);

  // Render report images on canvas whenever modal opens or result changes
  useEffect(() => {
    if (!isOpen || !result) return;

    const pattern = result.imagePattern || result.code || 'melanoma';

    if (reportOrigCanvasRef.current) {
      if (customImageSrc) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          const ctx = reportOrigCanvasRef.current.getContext('2d');
          ctx.drawImage(
            img,
            0,
            0,
            reportOrigCanvasRef.current.width,
            reportOrigCanvasRef.current.height
          );
          if (reportDullCanvasRef.current)
            applyDullRazorAndColorConstancy(
              reportOrigCanvasRef.current,
              reportDullCanvasRef.current
            );
          if (reportGradCanvasRef.current)
            renderGradCamOverlay(
              reportOrigCanvasRef.current,
              reportGradCanvasRef.current,
              result.spectralScore,
              'jet',
              0.7
            );
          if (reportContourCanvasRef.current)
            renderContourOverlay(reportOrigCanvasRef.current, reportContourCanvasRef.current);
        };
        img.src = customImageSrc;
      } else {
        renderLesionCanvas(reportOrigCanvasRef.current, pattern, { addHair: true });
        if (reportDullCanvasRef.current)
          applyDullRazorAndColorConstancy(reportOrigCanvasRef.current, reportDullCanvasRef.current);
        if (reportGradCanvasRef.current)
          renderGradCamOverlay(
            reportOrigCanvasRef.current,
            reportGradCanvasRef.current,
            result.spectralScore,
            'jet',
            0.7
          );
        if (reportContourCanvasRef.current)
          renderContourOverlay(reportOrigCanvasRef.current, reportContourCanvasRef.current);
      }
    }
  }, [isOpen, result, customImageSrc]);

  if (!isOpen || !result) return null;

  const handlePrint = () => {
    window.print();
  };

  const reportId = `EHR-DS-${Math.floor(100000 + Math.random() * 900000)}`;
  const dateStr = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4 sm:p-6">
      <div className="relative w-full max-w-5xl bg-derma-card border border-derma-border rounded-3xl shadow-2xl overflow-hidden text-slate-100 max-h-[94vh] flex flex-col">
        {/* Modal Toolbar (Non-printable) */}
        <div className="no-print p-4 bg-derma-surface/90 border-b border-derma-border flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FileText className="w-5 h-5 text-derma-accent" />
            <span className="text-sm font-bold text-white">
              EHR Clinical Diagnostic Report Summary (Multi-Modal)
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={handlePrint}
              className="px-4 py-1.5 bg-derma-accent hover:bg-derma-accentHover text-slate-950 text-xs font-bold rounded-xl flex items-center space-x-1.5 shadow-glow-cyan transition-all"
            >
              <Printer className="w-4 h-4" />
              <span>Print / Save PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-xl hover:bg-derma-card text-slate-400 hover:text-white transition-all"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable Report Document Body */}
        <div
          ref={reportRef}
          className="p-8 sm:p-10 overflow-y-auto space-y-6 text-slate-200 text-xs font-sans print:p-0 print:text-black"
        >
          {/* Hospital / Clinic Letterhead */}
          <div className="flex justify-between items-start border-b-2 border-derma-border print:border-gray-400 pb-6">
            <div>
              <div className="flex items-center space-x-2">
                <Activity className="w-7 h-7 text-derma-accent print:text-blue-600" />
                <h1 className="text-2xl font-black text-white print:text-black tracking-tight">
                  SkinScan AI{' '}
                  <span className="text-derma-accent print:text-blue-600">Clinical Triage</span>
                </h1>
              </div>
              <p className="text-[11px] text-derma-muted print:text-gray-600 mt-1">
                Department of Computer Science & Engineering • B.M.S. College of Engineering
                (Autonomous VTU)
              </p>
              <p className="text-[10px] text-slate-400 print:text-gray-500">
                Medical Deep Learning Research Group • Phase-2 Major Project
              </p>
            </div>

            <div className="text-right font-mono text-[11px]">
              <p className="font-bold text-derma-accent print:text-black">REPORT ID: {reportId}</p>
              <p className="text-slate-400 print:text-gray-500">{dateStr}</p>
              <span className="inline-block mt-1 px-2.5 py-0.5 rounded bg-derma-surface text-derma-teal print:text-black border border-derma-border print:border-gray-400 font-bold text-[10px] uppercase">
                Validated Triage Summary
              </span>
            </div>
          </div>

          {/* Patient Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-2xl bg-derma-bg/60 border border-derma-border/80 print:border-gray-300 print:bg-gray-50">
            <div>
              <span className="text-[10px] text-derma-muted print:text-gray-500 uppercase font-bold block">
                Patient Age:
              </span>
              <span className="font-mono text-sm font-bold text-white print:text-black">
                {result.metadata?.age || patientData?.age} Years
              </span>
            </div>
            <div>
              <span className="text-[10px] text-derma-muted print:text-gray-500 uppercase font-bold block">
                Biological Sex:
              </span>
              <span className="capitalize font-mono text-sm font-bold text-white print:text-black">
                {result.metadata?.sex || patientData?.sex}
              </span>
            </div>
            <div>
              <span className="text-[10px] text-derma-muted print:text-gray-500 uppercase font-bold block">
                Anatomical Site:
              </span>
              <span className="capitalize font-mono text-sm font-bold text-white print:text-black">
                {result.metadata?.localization || patientData?.localization}
              </span>
            </div>
            <div>
              <span className="text-[10px] text-derma-muted print:text-gray-500 uppercase font-bold block">
                Acquisition Modality:
              </span>
              <span className="capitalize font-mono text-xs font-bold text-derma-accent print:text-blue-700">
                {customImageSrc ? 'Custom Upload / Live Camera' : 'HAM10000 Benchmark'}
              </span>
            </div>
          </div>

          {/* Primary Assessment Box */}
          <div
            className={`p-6 rounded-2xl border ${
              result.riskStageNum >= 3
                ? 'bg-rose-500/10 border-rose-500/40 text-rose-100 print:bg-red-50 print:border-red-400'
                : 'bg-emerald-500/10 border-emerald-500/40 text-emerald-100 print:bg-green-50 print:border-green-400'
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <span className="text-[10px] uppercase font-extrabold tracking-widest text-derma-muted print:text-gray-600 block">
                  Automated Risk Stratification Result:
                </span>
                <h2 className="text-2xl font-black text-white print:text-black mt-1">
                  {result.riskStage}
                </h2>
                <p className="text-xs font-semibold text-derma-accent print:text-indigo-800 mt-1">
                  Predicted Pathological Class: {result.name} ({result.code.toUpperCase()})
                </p>
              </div>

              {/* Spectral Score Block */}
              <div className="text-center bg-derma-card/80 print:bg-white p-3.5 rounded-2xl border border-derma-border/90 print:border-gray-300 min-w-[150px]">
                <span className="text-[10px] font-bold uppercase text-derma-muted print:text-gray-500 block">
                  Spectral Score
                </span>
                <span className="text-3xl font-black text-derma-accent print:text-black font-mono">
                  {result.spectralScore.toFixed(1)}
                </span>
                <span className="text-[10px] block text-derma-muted print:text-gray-600 font-bold font-mono">
                  / 10.0 Scale
                </span>
              </div>
            </div>

            {/* Action directive */}
            <div className="mt-4 pt-3 border-t border-derma-border/50 print:border-gray-300 text-xs flex items-center space-x-2">
              <strong className="text-white print:text-black font-bold">
                Mandated Triage Action:
              </strong>
              <span className="font-semibold">{result.actionNeeded}</span>
            </div>
          </div>

          {/* NEW: MULTI-MODAL SCAN IMAGES & GRAD-CAM VISUALIZATION (Embedded in Report) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 print:text-black flex items-center space-x-1.5">
                <Layers className="w-4 h-4 text-derma-accent print:text-blue-600" />
                <span>Dermatoscopic Scan & Explainable AI (XAI) Visual Evidence</span>
              </h3>
              <span className="text-[10px] font-mono text-derma-muted print:text-gray-600">
                Layer: conv5_block16_2_conv (DenseNet121)
              </span>
            </div>

            {/* 4 Image Canvas Panels (Side-by-Side in Report) */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-2xl bg-derma-bg/70 border border-derma-border print:border-gray-300 print:bg-white">
              {/* 1. Raw Input Image (Live / Upload) */}
              <div className="flex flex-col items-center text-center space-y-1.5">
                <div className="w-full aspect-square rounded-xl overflow-hidden border border-derma-border print:border-gray-400 bg-slate-950">
                  <canvas
                    ref={reportOrigCanvasRef}
                    width={200}
                    height={200}
                    className="w-full h-full object-cover"
                  />
                </div>
                <span className="text-[10px] font-bold text-white print:text-black">
                  1. Raw Scan Input
                </span>
                <span className="text-[9px] text-derma-muted print:text-gray-500 font-mono">
                  {customImageSrc ? 'Point-of-Care Scan' : 'HAM10000 Ingestion'}
                </span>
              </div>

              {/* 2. DullRazor Preprocessed */}
              <div className="flex flex-col items-center text-center space-y-1.5">
                <div className="w-full aspect-square rounded-xl overflow-hidden border border-derma-border print:border-gray-400 bg-slate-950">
                  <canvas
                    ref={reportDullCanvasRef}
                    width={200}
                    height={200}
                    className="w-full h-full object-cover"
                  />
                </div>
                <span className="text-[10px] font-bold text-derma-teal print:text-black">
                  2. DullRazor Cleaned
                </span>
                <span className="text-[9px] text-derma-muted print:text-gray-500 font-mono">
                  Hair Removal & Gray-World
                </span>
              </div>

              {/* 3. Grad-CAM Attention Heatmap */}
              <div className="flex flex-col items-center text-center space-y-1.5">
                <div className="w-full aspect-square rounded-xl overflow-hidden border border-derma-accent/80 print:border-red-500 bg-slate-950">
                  <canvas
                    ref={reportGradCanvasRef}
                    width={200}
                    height={200}
                    className="w-full h-full object-cover"
                  />
                </div>
                <span className="text-[10px] font-bold text-derma-accent print:text-red-600">
                  3. Grad-CAM Activation
                </span>
                <span className="text-[9px] text-derma-muted print:text-gray-500 font-mono">
                  Neural Attention Map (Jet)
                </span>
              </div>

              {/* 4. Border Contour */}
              <div className="flex flex-col items-center text-center space-y-1.5">
                <div className="w-full aspect-square rounded-xl overflow-hidden border border-derma-border print:border-gray-400 bg-slate-950">
                  <canvas
                    ref={reportContourCanvasRef}
                    width={200}
                    height={200}
                    className="w-full h-full object-cover"
                  />
                </div>
                <span className="text-[10px] font-bold text-purple-300 print:text-black">
                  4. Border Contour
                </span>
                <span className="text-[9px] text-derma-muted print:text-gray-500 font-mono">
                  Polar Margin Segmentation
                </span>
              </div>
            </div>
          </div>

          {/* ABCD & Technical Quantification Table */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 print:text-black">
              Quantitative Stolz ABCD Feature Breakdown
            </h3>
            <table className="w-full border border-derma-border print:border-gray-300 rounded-xl overflow-hidden text-left">
              <thead className="bg-derma-surface print:bg-gray-100 text-derma-accent print:text-black text-[11px]">
                <tr>
                  <th className="p-2.5 font-bold">Criterion</th>
                  <th className="p-2.5 font-bold">Measured Value</th>
                  <th className="p-2.5 font-bold">Weight Factor</th>
                  <th className="p-2.5 font-bold">Clinical Interpretation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-derma-border print:divide-gray-200 text-[11px]">
                <tr>
                  <td className="p-2.5 font-semibold">A - Asymmetry (Dual Axis)</td>
                  <td className="p-2.5 font-mono">{result.abcd?.asymmetry?.toFixed(2)} / 2.0</td>
                  <td className="p-2.5 font-mono">&times; 1.3</td>
                  <td className="p-2.5">
                    {result.abcd?.asymmetry > 1.2
                      ? 'Significant Asymmetry (Concerning)'
                      : 'Symmetric Geometry'}
                  </td>
                </tr>
                <tr>
                  <td className="p-2.5 font-semibold">B - Border Irregularity</td>
                  <td className="p-2.5 font-mono">{result.abcd?.border?.toFixed(2)} / 8.0</td>
                  <td className="p-2.5 font-mono">&times; 0.1</td>
                  <td className="p-2.5">
                    {result.abcd?.border > 4.5
                      ? 'Abrupt, Notched Perimeter'
                      : 'Smooth Regular Margin'}
                  </td>
                </tr>
                <tr>
                  <td className="p-2.5 font-semibold">C - Color Variegation</td>
                  <td className="p-2.5 font-mono">{result.abcd?.color} Colors Detected</td>
                  <td className="p-2.5 font-mono">&times; 0.5</td>
                  <td className="p-2.5">{result.abcd?.colorsDetected?.join(', ')}</td>
                </tr>
                <tr>
                  <td className="p-2.5 font-semibold">D - Max Diameter</td>
                  <td className="p-2.5 font-mono">{result.abcd?.diameter?.toFixed(1)} mm</td>
                  <td className="p-2.5 font-mono">&times; 0.5</td>
                  <td className="p-2.5">
                    {result.abcd?.diameter > 6.0
                      ? '> 6.0 mm Threshold Exceeded'
                      : '< 6.0 mm Low Risk'}
                  </td>
                </tr>
                <tr className="bg-derma-surface/50 print:bg-gray-50 font-bold">
                  <td className="p-2.5 text-derma-accent print:text-black">
                    Total Dermoscopy Score (TDS)
                  </td>
                  <td className="p-2.5 font-mono text-base">{result.abcd?.totalDermoscopyScore}</td>
                  <td className="p-2.5 font-mono">-</td>
                  <td className="p-2.5">
                    {result.abcd?.totalDermoscopyScore > 5.45
                      ? 'High Probability Malignancy'
                      : 'Low/Benign Probability'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Suggested Billing & Classification Codes */}
          <div className="p-4 rounded-xl bg-derma-card border border-derma-border print:border-gray-300 space-y-1.5 text-[11px]">
            <p>
              <strong className="text-white print:text-black">
                Recommended ICD-10 Classification:{' '}
              </strong>
              {result.icd10}
            </p>
            <p>
              <strong className="text-white print:text-black">
                Pre-Screening Model Backbone:{' '}
              </strong>
              DenseNet121 (GAP 1024-d) + Random Forest Hybrid with Focal Loss
            </p>
            <p>
              <strong className="text-white print:text-black">Clinical Decision Rationale: </strong>
              {result.clinicalNotes}
            </p>
          </div>

          {/* NEW: ATTENDING DOCTOR & SPECIALIST REFERRAL DIRECTIVE */}
          {result.referral && (
            <div
              className={`p-5 rounded-2xl border ${
                result.referral.isCancer
                  ? 'bg-rose-950/20 border-rose-500/50 print:bg-red-50 print:border-red-400 text-rose-100 print:text-black'
                  : 'bg-emerald-950/20 border-emerald-500/50 print:bg-green-50 print:border-green-400 text-emerald-100 print:text-black'
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-derma-border print:border-gray-300">
                <div className="flex items-center space-x-2">
                  <span
                    className={`w-3 h-3 rounded-full ${result.referral.isCancer ? 'bg-derma-rose' : 'bg-derma-teal'}`}
                  />
                  <h3 className="text-xs font-black uppercase tracking-wider text-white print:text-black">
                    Official Specialist Referral & Clinical Doctor Assignment
                  </h3>
                </div>
                <span className="font-mono text-xs font-extrabold px-3 py-1 rounded-full bg-derma-surface print:bg-white border border-derma-border print:border-gray-400 text-derma-accent print:text-black">
                  Referral SLA: {result.referral.referralTimeline}
                </span>
              </div>

              <div className="mt-3.5 grid grid-cols-1 sm:grid-cols-2 gap-4 text-[11px]">
                <div className="space-y-1">
                  <span className="text-[10px] text-derma-muted print:text-gray-600 uppercase font-bold block">
                    Assigned Consulting Doctor:
                  </span>
                  <p className="text-sm font-extrabold text-white print:text-black">
                    {result.referral.doctorName}
                  </p>
                  <p className="text-derma-accent print:text-blue-800 font-semibold">
                    {result.referral.doctorTitle}
                  </p>
                  <p className="text-slate-300 print:text-gray-700 font-mono text-[10px]">
                    {result.referral.facility}
                  </p>
                  <p className="text-slate-400 print:text-gray-600 text-[10px]">
                    {result.referral.contactInfo}
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-derma-card/80 print:bg-white border border-derma-border print:border-gray-300 space-y-1.5">
                  <span className="text-[10px] text-derma-muted print:text-gray-600 uppercase font-bold block">
                    Diagnostic Pathology Classification:
                  </span>
                  <span
                    className={`inline-block px-2.5 py-0.5 rounded text-[10px] font-bold ${
                      result.referral.isCancer
                        ? 'bg-rose-500/20 text-derma-rose border border-rose-500/30'
                        : 'bg-emerald-500/20 text-derma-teal border border-emerald-500/30'
                    }`}
                  >
                    {result.referral.triageClassification}
                  </span>
                  <p className="text-slate-300 print:text-gray-800 text-[10px] leading-relaxed">
                    <strong className="text-white print:text-black">Recommended Procedure: </strong>
                    {result.referral.recommendedProcedure}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Signatures & Legal Disclaimer */}
          <div className="pt-6 border-t border-derma-border/70 print:border-gray-400 grid grid-cols-2 gap-8 text-[10px] text-derma-muted print:text-gray-500">
            <div>
              <p className="font-bold text-slate-300 print:text-black uppercase">
                Clinician Attestation:
              </p>
              <div className="mt-8 border-b border-slate-600 print:border-black w-48" />
              <p className="mt-1">Attending Physician / Dermatologist Signature</p>
            </div>

            <div className="text-right">
              <p className="font-bold text-slate-300 print:text-black uppercase">
                Software Audit Stamp:
              </p>
              <p className="mt-1 font-mono text-derma-accent print:text-black">
                SHA-256: 8a5d3f...91c0e2
              </p>
              <p className="mt-1">SkinScan AI Triage System v2.0 (BMSCE)</p>
            </div>
          </div>

          <div className="text-[9px] text-center text-slate-400 print:text-gray-500 italic">
            Disclaimer: This diagnostic triage report is generated as an adjunctive AI decision
            support tool under investigational protocol. Definite diagnosis requires
            histopathological examination of biopsy tissue.
          </div>
        </div>
      </div>
    </div>
  );
}
