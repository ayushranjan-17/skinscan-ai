import React, { useState, useRef, useEffect } from 'react';
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
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  // Mini canvas previews for benchmark cards
  const canvasRefs = useRef({});

  useEffect(() => {
    BENCHMARK_LESIONS.forEach((lesion) => {
      const canvas = canvasRefs.current[lesion.id];
      if (canvas) {
        renderLesionCanvas(canvas, lesion.imagePattern, {
          addHair: lesion.code === 'mel' || lesion.code === 'nv',
        });
      }
    });
  }, []);

  const [rawFile, setRawFile] = useState(null);

  // Handle local file upload
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setRawFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        const src = event.target.result;
        setCustomImageSrc(src);
        onSelectBenchmark?.(null); // Clear preset
        onSelectCustomImage?.(src, file);
      };
      reader.readAsDataURL(file);
    }
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

      // Stop stream
      const stream = videoRef.current.srcObject;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      setIsCameraActive(false);
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
            {activeInputTab === 'benchmarks' && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Select Validated Clinical Benchmark Case:
                  </span>
                  <span className="text-[11px] text-derma-accent">7 Classes (HAM10000)</span>
                </div>

                {/* Benchmark Carousel / Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-[340px] overflow-y-auto pr-1">
                  {BENCHMARK_LESIONS.map((lesion) => {
                    const isSelected = selectedBenchmark?.id === lesion.id;
                    return (
                      <div
                        key={lesion.id}
                        onClick={() => {
                          onSelectBenchmark(lesion);
                          setCustomImageSrc(null);
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
                              if (el) canvasRefs.current[lesion.id] = el;
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
              </div>
            )}

            {activeInputTab === 'upload' && (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-derma-border hover:border-derma-accent rounded-3xl p-8 text-center cursor-pointer transition-all bg-derma-bg/50 hover:bg-derma-surface/30 min-h-[300px] flex flex-col items-center justify-center"
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
                      Image loaded successfully. Click to replace.
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
                      Accepts high-resolution JPG, PNG, DICOM exports
                    </p>
                    <span className="mt-4 px-4 py-1.5 rounded-xl bg-derma-card border border-derma-border text-xs font-semibold text-derma-accent">
                      Browse Local File
                    </span>
                  </>
                )}
              </div>
            )}

            {activeInputTab === 'camera' && (
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
            )}
          </div>
        )}

        {/* Right Col: Patient Metadata Form (conditionally hidden in step-by-step mode) */}
        {!hidePatientForm && (
          <div
            className={`${hideImageInput ? 'lg:col-span-12' : 'lg:col-span-5'} bg-derma-bg/60 p-5 rounded-2xl border border-derma-border/80 flex flex-col justify-between space-y-4`}
          >
            <div>
              <div className="flex items-center space-x-2 pb-3 border-b border-derma-border/50">
                <User className="w-4 h-4 text-derma-accent" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                  Patient Metadata & Clinical Prior
                </h3>
              </div>

              <div className="mt-4 space-y-3.5 text-xs">
                {/* Age & Sex Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-400 font-medium mb-1">
                      Patient Age (Years)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="105"
                      value={patientData.age}
                      onChange={(e) => onUpdatePatientData({ age: Number(e.target.value) })}
                      className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white font-mono focus:border-derma-accent focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-400 font-medium mb-1">Biological Sex</label>
                    <select
                      value={patientData.sex}
                      onChange={(e) => onUpdatePatientData({ sex: e.target.value })}
                      className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white focus:border-derma-accent focus:outline-none"
                    >
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="other">Other / Unknown</option>
                    </select>
                  </div>
                </div>

                {/* Anatomical Site */}
                <div>
                  <label className="block text-slate-400 font-medium mb-1">
                    Anatomical Site / Localization
                  </label>
                  <select
                    value={patientData.localization}
                    onChange={(e) => onUpdatePatientData({ localization: e.target.value })}
                    className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white focus:border-derma-accent focus:outline-none capitalize"
                  >
                    <option value="back">Back (High UV Exposure)</option>
                    <option value="face">Face / Head</option>
                    <option value="scalp">Scalp / Neck</option>
                    <option value="chest">Anterior Chest</option>
                    <option value="abdomen">Abdomen</option>
                    <option value="upper extremity">Upper Extremity / Arms</option>
                    <option value="lower extremity">Lower Extremity / Legs</option>
                    <option value="trunk">Trunk</option>
                  </select>
                </div>

                {/* Lesion Duration & History */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-400 font-medium mb-1">
                      Observed (Months)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={patientData.durationMonths}
                      onChange={(e) =>
                        onUpdatePatientData({ durationMonths: Number(e.target.value) })
                      }
                      className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white font-mono focus:border-derma-accent focus:outline-none"
                    />
                  </div>

                  <div className="flex flex-col justify-end">
                    <label className="flex items-center space-x-2 p-2 bg-derma-card rounded-xl border border-derma-border cursor-pointer hover:border-derma-accent">
                      <input
                        type="checkbox"
                        checked={patientData.historyOfMelanoma}
                        onChange={(e) =>
                          onUpdatePatientData({ historyOfMelanoma: e.target.checked })
                        }
                        className="rounded bg-derma-surface text-derma-accent focus:ring-derma-accent"
                      />
                      <span className="text-[11px] text-slate-300 font-medium">
                        Prior Melanoma Hx
                      </span>
                    </label>
                  </div>
                </div>

                {/* Doctor Routing Preference */}
                <div>
                  <label className="block text-slate-400 font-medium mb-1">
                    Doctor Referral Routing Mode
                  </label>
                  <select
                    value={patientData.doctorPreference || 'auto'}
                    onChange={(e) => onUpdatePatientData({ doctorPreference: e.target.value })}
                    className="w-full bg-derma-card border border-derma-border rounded-xl px-3 py-2 text-white focus:border-derma-accent focus:outline-none"
                  >
                    <option value="auto">⚡ Auto-Route Best Doctor Based on AI Pathology</option>
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
            </div>

            {/* Trigger Triage Execution Button */}
            {!hideAnalyzeButton && (
              <div className="pt-2">
                <button
                  onClick={() => onAnalyze(customImageSrc, rawFile)}
                  disabled={isAnalyzing}
                  className={`w-full py-3.5 rounded-2xl text-xs font-extrabold uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-glow-cyan ${
                    isAnalyzing
                      ? 'bg-derma-border text-slate-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-derma-accent via-cyan-300 to-derma-blue text-slate-950 hover:opacity-95 active:scale-[0.98]'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-derma-accent" />
                      <span>Processing Multi-Modal Pipeline...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Execute AI Triage & Risk Stratification</span>
                    </>
                  )}
                </button>
                <p className="text-[10px] text-center text-derma-muted mt-2">
                  Runs DullRazor hair removal • DenseNet121 extraction • Two-Step Hierarchical Logic
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
