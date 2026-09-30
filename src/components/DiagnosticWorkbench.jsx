import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  Camera,
  Sparkles,
  User,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Info,
  ChevronDown,
} from 'lucide-react';
import { BENCHMARK_LESIONS } from '../data/benchmarks';
import { renderLesionCanvas } from '../utils/imageProcessing';

export default function DiagnosticWorkbench({
  selectedBenchmark,
  onSelectBenchmark,
  onAnalyze,
  patientData,
  onUpdatePatientData,
  isAnalyzing,
  hidePatientForm = false,
  hideAnalyzeButton = false,
  hideImageInput = false,
  customImageSrc: parentCustomImageSrc = null,
  onSelectCustomImage = null,
}) {
  const [activeInputTab, setActiveInputTab] = useState('benchmarks'); // 'benchmarks' | 'upload' | 'camera'
  const [customImageSrc, setCustomImageSrc] = useState(parentCustomImageSrc);
  const [rawFile, setRawFile] = useState(null);
  const [skipMetadata, setSkipMetadata] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  // Mini canvas previews for benchmark cards
  const canvasRefs = useRef({});

  // Sync parent customImageSrc when it changes
  useEffect(() => {
    setCustomImageSrc(parentCustomImageSrc);
  }, [parentCustomImageSrc]);

  const hasImage = Boolean(customImageSrc || selectedBenchmark || rawFile);

  const drawBenchmarkCanvas = useCallback((canvas, lesion) => {
    if (!canvas || !lesion) return;
    renderLesionCanvas(canvas, lesion.imagePattern, {
      addHair: lesion.code === 'mel' || lesion.code === 'nv',
    });
  }, []);

  const drawAllBenchmarkCanvases = useCallback(() => {
    BENCHMARK_LESIONS.forEach((lesion) => {
      const canvas = canvasRefs.current[lesion.id];
      if (canvas) {
        drawBenchmarkCanvas(canvas, lesion);
      }
    });
  }, [drawBenchmarkCanvas]);

  useEffect(() => {
    drawAllBenchmarkCanvases();
  }, [drawAllBenchmarkCanvases, activeInputTab]);

  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const processSelectedFile = (file) => {
    if (!file) return;
    if (!file.type || !file.type.startsWith('image/')) {
      setUploadError('Invalid file format. Please upload a valid JPG, PNG, or WebP image.');
      return;
    }
    setUploadError(null);
    setRawFile(file);
    const reader = new FileReader();
    reader.onload = (event) => {
      const src = event.target.result;
      setCustomImageSrc(src);
      onSelectBenchmark?.(null); // Clear preset
      onSelectCustomImage?.(src, file);
    };
    reader.readAsDataURL(file);
  };

  // Handle local file upload
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    processSelectedFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer?.files?.[0];
    processSelectedFile(file);
  };

  // Handle webcam stream
  const startCamera = async () => {
    try {
      setIsCameraActive(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Unable to access device camera. Please upload an image instead.');
      setIsCameraActive(false);
    }
  };

  const captureCameraPhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth || 400;
      canvas.height = videoRef.current.videoHeight || 400;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      const photoUrl = canvas.toDataURL('image/jpeg');
      setCustomImageSrc(photoUrl);
      canvas.toBlob((blob) => {
        setRawFile(blob);
        onSelectBenchmark?.(null);
        onSelectCustomImage?.(photoUrl, blob);
      }, 'image/jpeg');

      stopCamera();
    }
  };

  return (
    <div className="glass-panel p-6 rounded-3xl border border-derma-border shadow-2xl relative">
      {/* Header */}
      {!hideImageInput && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-derma-border/70">
          <div>
            <div className="flex items-center space-x-2">
              <span className="p-2 rounded-xl bg-derma-blue/20 text-derma-accent border border-derma-border">
                <Sparkles className="w-5 h-5" />
              </span>
              <h2 className="text-xl font-bold text-white font-sans">
                Clinical Ingestion & Patient Context
              </h2>
            </div>
            <p className="text-xs text-derma-muted mt-1">
              Provide a dermoscopic scan and multi-modal patient metadata to execute two-tier risk
              triage.
            </p>
          </div>

          {/* Input Switcher Tabs */}
          <div className="flex items-center space-x-1 bg-derma-bg/90 p-1.5 rounded-xl border border-derma-border text-xs font-semibold">
            <button
              onClick={() => {
                stopCamera();
                setActiveInputTab('benchmarks');
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                activeInputTab === 'benchmarks'
                  ? 'bg-derma-surface text-derma-accent border border-derma-border font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              HAM10000 Benchmarks
            </button>
            <button
              onClick={() => {
                stopCamera();
                setActiveInputTab('upload');
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                activeInputTab === 'upload'
                  ? 'bg-derma-surface text-derma-accent border border-derma-border font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Upload File
            </button>
            <button
              onClick={() => {
                setActiveInputTab('camera');
                startCamera();
              }}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                activeInputTab === 'camera'
                  ? 'bg-derma-surface text-derma-accent border border-derma-border font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Live Camera
            </button>
          </div>
        </div>
      )}

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Image Input Area (conditionally hidden in step-by-step mode) */}
        {!hideImageInput && (
          <div className={`${hidePatientForm ? 'lg:col-span-12' : 'lg:col-span-7'} space-y-4`}>
            {/* Benchmarks Tab */}
            <div className={activeInputTab === 'benchmarks' ? 'block' : 'hidden'}>
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Select Validated Clinical Benchmark Case:
                  </span>
                  <span className="text-[11px] text-derma-accent font-semibold">7 Classes (HAM10000)</span>
                </div>

                {/* Benchmark Carousel / Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-[340px] overflow-y-auto pr-1">
                  {BENCHMARK_LESIONS.map((lesion) => {
                    const isSelected = selectedBenchmark?.id === lesion.id;
                    return (
                      <div
                        key={lesion.id}
                        onClick={() => {
                          onSelectBenchmark?.(lesion);
                          setCustomImageSrc(null);
                          setRawFile(null);
                        }}
                        className={`cursor-pointer p-2.5 rounded-2xl border transition-all flex flex-col items-center text-center ${
                          isSelected
                            ? 'border-derma-accent bg-derma-surface/90 shadow-glow-cyan scale-[1.02]'
                            : 'border-derma-border/70 bg-derma-card/60 hover:border-slate-500 hover:bg-derma-card'
                        }`}
                      >
                        <div className="relative w-16 h-16 rounded-xl overflow-hidden shadow-inner border border-derma-border/90 bg-slate-950">
                          <canvas
                            ref={(el) => {
                              if (el) {
                                canvasRefs.current[lesion.id] = el;
                                drawBenchmarkCanvas(el, lesion);
                              }
                            }}
                            width={100}
                            height={100}
                            className="w-full h-full object-cover"
                          />
                          {isSelected && (
                            <div className="absolute inset-0 bg-derma-accent/20 flex items-center justify-center">
                              <CheckCircle2 className="w-5 h-5 text-derma-accent bg-derma-bg rounded-full" />
                            </div>
                          )}
                        </div>

                        <span className="text-xs font-bold text-white mt-2 truncate w-full">
                          {lesion.shortName}
                        </span>

                        <div className="mt-1 flex items-center space-x-1">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              lesion.riskStageNum === 4
                                ? 'bg-derma-rose'
                                : lesion.riskStageNum === 3
                                  ? 'bg-derma-amber'
                                  : lesion.riskStageNum === 2
                                    ? 'bg-derma-yellow'
                                    : 'bg-derma-teal'
                            }`}
                          />
                          <span className="text-[10px] text-derma-muted font-mono">
                            Score: {lesion.spectralScore}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Selected Benchmark Detail Preview Banner */}
                {selectedBenchmark && (
                  <div className="mt-3.5 p-3.5 rounded-2xl bg-derma-surface/80 border border-derma-accent/40 flex items-center justify-between shadow-sm">
                    <div className="flex items-center space-x-3 min-w-0">
                      <div className="relative w-12 h-12 rounded-xl overflow-hidden border border-derma-accent shadow-sm bg-slate-950 flex-shrink-0">
                        <canvas
                          ref={(el) => {
                            if (el && selectedBenchmark) {
                              drawBenchmarkCanvas(el, selectedBenchmark);
                            }
                          }}
                          width={120}
                          height={120}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs font-bold text-white truncate">
                            {selectedBenchmark.name}
                          </span>
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-derma-accent/20 text-derma-accent border border-derma-accent/30 uppercase flex-shrink-0">
                            {selectedBenchmark.code}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-300 mt-0.5 truncate">
                          {selectedBenchmark.category} • Baseline Risk: <span className="font-mono text-derma-accent font-bold">{selectedBenchmark.spectralScore}</span> / 10
                        </p>
                      </div>
                    </div>
                    <div className="flex-shrink-0 ml-2">
                      <span className="inline-flex items-center px-2 py-1 rounded-lg text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                        ✓ Selected Case
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Upload Tab */}
            <div className={activeInputTab === 'upload' ? 'block' : 'hidden'}>
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragEnter={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-3xl p-8 text-center cursor-pointer transition-all min-h-[300px] flex flex-col items-center justify-center ${
                  isDragging
                    ? 'border-derma-accent bg-derma-accent/10 scale-[1.01]'
                    : 'border-derma-border hover:border-derma-accent bg-derma-bg/50 hover:bg-derma-surface/30'
                }`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                />
                {customImageSrc ? (
                  <div className="space-y-3">
                    <img
                      src={customImageSrc}
                      alt="Custom Upload"
                      className="w-36 h-36 rounded-2xl object-cover mx-auto border-2 border-derma-accent shadow-glow-cyan"
                    />
                    <p className="text-xs text-derma-accent font-semibold">
                      Image loaded successfully. Click or drop another file to replace.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="p-4 rounded-2xl bg-derma-surface text-derma-accent mb-3 shadow-inner">
                      <Upload className="w-8 h-8" />
                    </div>
                    <p className="text-sm font-bold text-white">
                      Click or drag & drop dermoscopic photo
                    </p>
                    <p className="text-xs text-derma-muted mt-1">
                      Accepts high-resolution JPG, PNG, WebP exports
                    </p>
                    <span className="mt-4 px-4 py-1.5 rounded-xl bg-derma-card border border-derma-border text-xs font-semibold text-derma-accent">
                      Browse Local File
                    </span>
                  </>
                )}
                {uploadError && (
                  <p className="mt-3 text-xs text-derma-rose font-semibold bg-rose-950/40 border border-rose-500/40 px-3 py-1.5 rounded-xl">
                    {uploadError}
                  </p>
                )}
              </div>
            </div>

            {/* Camera Tab */}
            <div className={activeInputTab === 'camera' ? 'block' : 'hidden'}>
              <div className="relative rounded-3xl overflow-hidden border border-derma-border bg-black min-h-[300px] flex items-center justify-center">
                {isCameraActive ? (
                  <div className="relative w-full h-[300px]">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-full object-cover"
                    />
                    {/* Viewfinder Target Circle */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-48 h-48 rounded-full border-2 border-dashed border-derma-accent shadow-glow-cyan animate-pulse flex items-center justify-center">
                        <span className="text-[10px] text-derma-accent font-mono bg-derma-bg/70 px-2 py-0.5 rounded">
                          Align Lesion Here
                        </span>
                      </div>
                    </div>
                    <div className="absolute bottom-4 inset-x-0 flex justify-center">
                      <button
                        onClick={captureCameraPhoto}
                        className="px-6 py-2.5 bg-derma-accent hover:bg-derma-accentHover text-slate-950 font-bold rounded-2xl text-xs flex items-center space-x-2 shadow-glow-cyan"
                      >
                        <Camera className="w-4 h-4" />
                        <span>Capture Lesion Scan</span>
                      </button>
                    </div>
                  </div>
                ) : customImageSrc ? (
                  <div className="p-6 text-center space-y-3">
                    <img
                      src={customImageSrc}
                      alt="Captured"
                      className="w-36 h-36 rounded-2xl object-cover mx-auto border border-derma-accent"
                    />
                    <button
                      onClick={startCamera}
                      className="px-4 py-1.5 bg-derma-surface text-derma-accent text-xs font-semibold rounded-xl border border-derma-border"
                    >
                      Retake Photo
                    </button>
                  </div>
                ) : (
                  <div className="p-6 text-center">
                    <Camera className="w-8 h-8 text-derma-muted mx-auto mb-2" />
                    <p className="text-xs text-derma-muted">Camera stream ready</p>
                    <button
                      onClick={startCamera}
                      className="mt-3 px-4 py-2 bg-derma-accent text-slate-950 text-xs font-bold rounded-xl"
                    >
                      Activate Camera
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Right Col: Patient Metadata Form (conditionally hidden in step-by-step mode) */}
        {!hidePatientForm && (
          <div
            className={`${hideImageInput ? 'lg:col-span-12' : 'lg:col-span-5'} bg-derma-bg/60 p-5 rounded-2xl border border-derma-border/80 flex flex-col justify-between space-y-4`}
          >
            <div>
              {/* Active Target Banner in Patient Info (Step 2) */}
              {hideImageInput && (
                <div className="mb-4 p-3.5 rounded-2xl bg-derma-surface/80 border border-derma-border flex items-center justify-between">
                  <div className="flex items-center space-x-3 min-w-0">
                    <div className="w-11 h-11 rounded-xl overflow-hidden border border-derma-accent/60 bg-slate-950 flex-shrink-0">
                      {customImageSrc ? (
                        <img src={customImageSrc} alt="Lesion" className="w-full h-full object-cover" />
                      ) : selectedBenchmark ? (
                        <canvas
                          ref={(el) => {
                            if (el && selectedBenchmark) {
                              drawBenchmarkCanvas(el, selectedBenchmark);
                            }
                          }}
                          width={100}
                          height={100}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-derma-muted text-xs">?</div>
                      )}
                    </div>
                    <div className="min-w-0">
                      <p className="text-[10px] text-derma-muted font-bold uppercase tracking-wider">
                        Active Diagnostic Ingestion Target:
                      </p>
                      <p className="text-xs font-bold text-white truncate">
                        {customImageSrc
                          ? "Custom Dermoscopic Upload / Camera"
                          : selectedBenchmark
                            ? selectedBenchmark.name
                            : "No Lesion Image Selected"}
                      </p>
                    </div>
                  </div>
                  {selectedBenchmark && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-derma-accent/20 text-derma-accent border border-derma-accent/30 uppercase flex-shrink-0 ml-2">
                      HAM10000: {selectedBenchmark.code.toUpperCase()}
                    </span>
                  )}
                </div>
              )}

              <div className="flex items-center justify-between pb-3 border-b border-derma-border/50">
                <div className="flex items-center space-x-2">
                  <User className="w-4 h-4 text-derma-accent" />
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    Patient Metadata & Prior
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const nextSkip = !skipMetadata;
                    setSkipMetadata(nextSkip);
                    if (nextSkip) {
                      onUpdatePatientData?.({
                        age: 50,
                        sex: 'other',
                        localization: 'unknown',
                        durationMonths: 6,
                        historyOfMelanoma: false,
                      });
                    }
                  }}
                  className={`text-[11px] font-semibold px-2.5 py-1 rounded-lg border transition-all ${
                    skipMetadata
                      ? "bg-derma-accent/20 border-derma-accent text-derma-accent font-bold"
                      : "bg-derma-card border-derma-border text-slate-400 hover:text-white"
                  }`}
                >
                  {skipMetadata ? "✓ Skipped (Image Only)" : "Skip Patient Info"}
                </button>
              </div>

              {skipMetadata ? (
                <div className="mt-4 p-4 rounded-2xl bg-derma-surface/60 border border-derma-accent/30 text-center space-y-2.5">
                  <div className="w-8 h-8 rounded-full bg-derma-accent/20 text-derma-accent flex items-center justify-center mx-auto">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <p className="text-xs font-bold text-white">⚡ Image-Only Mode Active</p>
                  <p className="text-[11px] text-derma-muted leading-relaxed">
                    AI classification will evaluate the lesion purely from dermatoscopic visual features (DenseNet121 + DullRazor + ABCD scoring). Patient demographic records are skipped.
                  </p>
                  <button
                    type="button"
                    onClick={() => setSkipMetadata(false)}
                    className="text-[11px] text-derma-accent hover:underline font-semibold block mx-auto pt-1"
                  >
                    ← Click to enter patient details
                  </button>
                </div>
              ) : (

              <div className="mt-4 space-y-3.5 text-xs">
                {/* Clinical Presentation Category (Mole vs Wound) */}
                <div>
                  <label className="block text-slate-400 font-medium mb-1.5">
                    Clinical Presentation Type
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => onUpdatePatientData?.({ lesionType: 'mole' })}
                      className={`p-2 rounded-xl border text-center transition-all flex items-center justify-center space-x-1.5 ${
                        patientData?.lesionType !== 'wound'
                          ? 'bg-derma-accent/20 border-derma-accent text-white font-bold shadow-sm shadow-derma-accent/20'
                          : 'bg-derma-card border-derma-border text-slate-400 hover:text-white'
                      }`}
                    >
                      <span>🔘 Pigmented Mole / Spot</span>
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        onUpdatePatientData?.({
                          lesionType: 'wound',
                          durationMonths: 1,
                          historyOfMelanoma: false,
                        })
                      }
                      className={`p-2 rounded-xl border text-center transition-all flex items-center justify-center space-x-1.5 ${
                        patientData?.lesionType === 'wound'
                          ? 'bg-amber-500/25 border-amber-500 text-amber-300 font-bold shadow-sm shadow-amber-500/20'
                          : 'bg-derma-card border-derma-border text-slate-400 hover:text-white'
                      }`}
                    >
                      <span>🩹 Acute Wound / Scab / Cut</span>
                    </button>
                  </div>
                </div>

                {/* Age & Sex Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-400 font-medium mb-1">
                      Patient Age (Years)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={patientData?.age ?? ''}
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === '') {
                          onUpdatePatientData?.({ age: '' });
                        } else {
                          const parsed = parseInt(raw, 10);
                          if (!Number.isNaN(parsed)) {
                            onUpdatePatientData?.({ age: Math.min(120, Math.max(0, parsed)) });
                          }
                        }
                      }}
                      onBlur={() => {
                        const num = Number(patientData?.age);
                        if (!num || num < 1) onUpdatePatientData?.({ age: 1 });
                        else if (num > 120) onUpdatePatientData?.({ age: 120 });
                      }}
                      className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white font-mono focus:border-derma-accent focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-400 font-medium mb-1">Biological Sex</label>
                    <select
                      value={patientData?.sex ?? 'male'}
                      onChange={(e) => onUpdatePatientData?.({ sex: e.target.value })}
                      className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white focus:border-derma-accent focus:outline-none"
                    >
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="other">Unknown / Other</option>
                    </select>
                  </div>
                </div>

                {/* Localization */}
                <div>
                  <label className="block text-slate-400 font-medium mb-1">
                    Anatomic Lesion Site
                  </label>
                  <select
                    value={patientData?.localization ?? 'back'}
                    onChange={(e) => onUpdatePatientData?.({ localization: e.target.value })}
                    className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white focus:border-derma-accent focus:outline-none"
                  >
                    <option value="back">Back / Torso (High Sun Exposure)</option>
                    <option value="lower extremity">Lower Extremity (Leg / Ankle / Foot)</option>
                    <option value="trunk">Trunk / Abdomen / Chest</option>
                    <option value="upper extremity">Upper Extremity (Arm / Hand)</option>
                    <option value="abdomen">Abdomen</option>
                    <option value="face">Face / Scalp / Ear (UV Critical)</option>
                    <option value="chest">Chest</option>
                    <option value="neck">Neck</option>
                    <option value="unknown">Other / Unspecified</option>
                  </select>
                </div>

                {/* Duration & History */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-400 font-medium mb-1">
                      Lesion Duration (Mo)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="600"
                      value={patientData?.durationMonths ?? ''}
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === '') {
                          onUpdatePatientData?.({ durationMonths: '' });
                        } else {
                          const parsed = parseInt(raw, 10);
                          if (!Number.isNaN(parsed)) {
                            onUpdatePatientData?.({
                              durationMonths: Math.min(600, Math.max(0, parsed)),
                            });
                          }
                        }
                      }}
                      onBlur={() => {
                        const num = Number(patientData?.durationMonths);
                        if (!num || num < 1) onUpdatePatientData?.({ durationMonths: 1 });
                        else if (num > 600) onUpdatePatientData?.({ durationMonths: 600 });
                      }}
                      className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white font-mono focus:border-derma-accent focus:outline-none"
                    />
                  </div>

                  <div className="flex flex-col justify-end">
                    <label className="flex items-center space-x-2 bg-derma-card p-2.5 rounded-xl border border-derma-border cursor-pointer hover:border-slate-500">
                      <input
                        type="checkbox"
                        checked={patientData?.historyOfMelanoma ?? false}
                        onChange={(e) =>
                          onUpdatePatientData?.({ historyOfMelanoma: e.target.checked })
                        }
                        className="rounded border-derma-border text-derma-accent focus:ring-0 focus:ring-offset-0 bg-derma-bg"
                      />
                      <span className="text-[11px] text-slate-300 font-medium leading-tight">
                        Prior Melanoma History
                      </span>
                    </label>
                  </div>
                </div>

                {/* Doctor Triage Preference */}
                <div>
                  <label className="block text-slate-400 font-medium mb-1">
                    Doctor Specialty Routing Preference
                  </label>
                  <select
                    value={patientData?.doctorPreference ?? 'auto'}
                    onChange={(e) => onUpdatePatientData?.({ doctorPreference: e.target.value })}
                    className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white focus:border-derma-accent focus:outline-none text-xs"
                  >
                    <option value="auto">
                      ⚡ Automated AI Recommendation (Based on Risk Tier)
                    </option>
                    <option value="pcp">
                      👨‍⚕️ Local Primary Care / Family Doctor (Routine/Wounds)
                    </option>
                    <option value="derm">
                      🔬 Consultant Clinical Dermatologist (Skin Lesions)
                    </option>
                    <option value="onco">
                      🏥 Cutaneous Surgical Oncologist (High-Risk/Cancer)
                    </option>
                    <option value="mohs">🩺 Mohs Micrographic Surgeon (Facial/Keratinocyte)</option>
                  </select>
                </div>
              </div>
              )}
            </div>

            {/* Trigger Triage Execution Button */}
            {!hideAnalyzeButton && (
              <div className="pt-2">
                <button
                  onClick={() => {
                    if (skipMetadata) {
                      onUpdatePatientData?.({
                        age: 50,
                        sex: "other",
                        localization: "unknown",
                        durationMonths: 6,
                        historyOfMelanoma: false,
                      });
                    }
                    onAnalyze?.(customImageSrc, rawFile);
                  }}
                  disabled={isAnalyzing || !hasImage}
                  className={`w-full py-3.5 rounded-2xl text-xs font-extrabold uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-glow-cyan ${
                    isAnalyzing || !hasImage
                      ? 'bg-derma-border text-slate-500 cursor-not-allowed opacity-50'
                      : 'bg-gradient-to-r from-derma-accent via-cyan-300 to-derma-blue text-slate-950 hover:opacity-95 active:scale-[0.98]'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-derma-accent" />
                      <span>Processing Multi-Modal Pipeline...</span>
                    </>
                  ) : !hasImage ? (
                    <>
                      <Upload className="w-4 h-4" />
                      <span>Upload or Select an Image First</span>
                    </>
                  ) : skipMetadata ? (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Execute AI Triage (Image Only)</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Execute AI Triage & Risk Stratification</span>
                    </>
                  )}
                </button>
                <p className="text-[10px] text-center text-derma-muted mt-2">
                  {!hasImage
                    ? "A dermoscopic photo is required before running AI analysis"
                    : skipMetadata
                      ? "Image-only mode • Pure visual DenseNet121 + DullRazor + ABCD rule"
                      : "Runs DullRazor hair removal • DenseNet121 extraction • Two-Step Hierarchical Logic"}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
