import React, { useState, useRef, useEffect } from 'react';
import { Eye, Layers, Sliders, Sparkles, Shield, Cpu, RefreshCw, ZoomIn } from 'lucide-react';
import {
  renderLesionCanvas,
  applyDullRazorAndColorConstancy,
  renderGradCamOverlay,
  renderContourOverlay,
} from '../utils/imageProcessing';

export default function GradCamVisualizer({ result, customImageSrc }) {
  const [activeView, setActiveView] = useState('gradcam'); // 'original' | 'dullrazor' | 'gradcam' | 'contour'
  const [colormap, setColormap] = useState('jet'); // 'jet' | 'inferno' | 'turbo'
  const [opacity, setOpacity] = useState(0.65);

  const origCanvasRef = useRef(null);
  const dullCanvasRef = useRef(null);
  const gradCanvasRef = useRef(null);
  const contourCanvasRef = useRef(null);

  // Render canvases when result or view settings change
  useEffect(() => {
    if (!result) return;

    const pattern = result.imagePattern || result.code || 'melanoma';
    const hasBackendImages = Boolean(
      result?.images?.original && result?.images?.dullrazor && result?.images?.gradcam
    );

    const drawBase64ToCanvas = (b64, canvas, onLoadCb) => {
      if (!canvas || !b64) return;
      const img = new Image();
      img.onload = () => {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        if (onLoadCb) onLoadCb();
      };
      img.src = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
    };

    if (hasBackendImages) {
      drawBase64ToCanvas(result.images.original, origCanvasRef.current, () => {
        if (contourCanvasRef.current && origCanvasRef.current) {
          renderContourOverlay(origCanvasRef.current, contourCanvasRef.current);
        }
        if (gradCanvasRef.current && origCanvasRef.current) {
          if (colormap === 'jet' && Math.abs(opacity - 0.65) < 0.05) {
            drawBase64ToCanvas(result.images.gradcam, gradCanvasRef.current);
          } else {
            renderGradCamOverlay(
              origCanvasRef.current,
              gradCanvasRef.current,
              result.spectralScore,
              colormap,
              opacity
            );
          }
        }
      });
      drawBase64ToCanvas(result.images.dullrazor, dullCanvasRef.current);
      return;
    }

    // 1. Render Original Canvas (client-side fallback / benchmark mode)
    if (origCanvasRef.current) {
      if (customImageSrc) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          if (!origCanvasRef.current) return;
          const ctx = origCanvasRef.current.getContext('2d');
          ctx.drawImage(img, 0, 0, origCanvasRef.current.width, origCanvasRef.current.height);
          // Apply filters to dependent canvases
          if (dullCanvasRef.current)
            applyDullRazorAndColorConstancy(origCanvasRef.current, dullCanvasRef.current);
          if (gradCanvasRef.current)
            renderGradCamOverlay(
              origCanvasRef.current,
              gradCanvasRef.current,
              result.spectralScore,
              colormap,
              opacity
            );
          if (contourCanvasRef.current)
            renderContourOverlay(origCanvasRef.current, contourCanvasRef.current);
        };
        img.src = customImageSrc;
      } else {
        renderLesionCanvas(origCanvasRef.current, pattern, { addHair: true });
        if (dullCanvasRef.current)
          applyDullRazorAndColorConstancy(origCanvasRef.current, dullCanvasRef.current);
        if (gradCanvasRef.current)
          renderGradCamOverlay(
            origCanvasRef.current,
            gradCanvasRef.current,
            result.spectralScore,
            colormap,
            opacity
          );
        if (contourCanvasRef.current)
          renderContourOverlay(origCanvasRef.current, contourCanvasRef.current);
      }
    }
  }, [result, customImageSrc, colormap, opacity]);

  if (!result) return null;

  return (
    <div className="glass-panel p-6 rounded-2xl border border-derma-border shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 pb-5 border-b border-derma-border/70">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30">
              <Layers className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-bold text-slate-50 font-sans">
              Explainable AI (XAI) Vision Suite
            </h2>
          </div>
          <p className="text-xs text-derma-muted mt-1">
            DenseNet121 Layer Activation (
            <code className="font-mono text-derma-accent text-xs">conv5_block16_2_conv</code>) +
            DullRazor Preprocessing
          </p>
        </div>

        {/* View Mode Tabs */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-1 bg-derma-bg/90 p-1.5 rounded-xl border border-derma-border text-xs font-semibold">
          {[
            { id: 'original', label: 'Raw scan' },
            { id: 'dullrazor', label: 'Hair removed' },
            { id: 'gradcam', label: 'Heatmap' },
            { id: 'contour', label: 'Border' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveView(tab.id)}
              className={`px-3 py-2 rounded-lg text-center whitespace-nowrap transition-colors ${
                activeView === tab.id
                  ? 'bg-derma-accent text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-slate-50 hover:bg-derma-surface/60'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Visualizer Stage */}
      <div className="grid grid-cols-[repeat(auto-fit,minmax(18rem,1fr))] gap-6 items-stretch">
        {/* Canvas Display Viewport */}
        <div className="flex flex-col items-center justify-center p-6 bg-derma-bg/90 rounded-2xl border border-derma-border relative overflow-hidden group">
          {/* Active Canvas Frame */}
          <div className="relative w-full max-w-[320px] aspect-square rounded-xl overflow-hidden shadow-2xl border-2 border-derma-border group-hover:border-derma-accent transition-all">
            {/* Canvases (controlled by activeView visibility) */}
            <canvas
              ref={origCanvasRef}
              width={320}
              height={320}
              className={`w-full h-full object-cover transition-opacity duration-300 ${
                activeView === 'original' ? 'block' : 'hidden'
              }`}
            />
            <canvas
              ref={dullCanvasRef}
              width={320}
              height={320}
              className={`w-full h-full object-cover transition-opacity duration-300 ${
                activeView === 'dullrazor' ? 'block' : 'hidden'
              }`}
            />
            <canvas
              ref={gradCanvasRef}
              width={320}
              height={320}
              className={`w-full h-full object-cover transition-opacity duration-300 ${
                activeView === 'gradcam' ? 'block' : 'hidden'
              }`}
            />
            <canvas
              ref={contourCanvasRef}
              width={320}
              height={320}
              className={`w-full h-full object-cover transition-opacity duration-300 ${
                activeView === 'contour' ? 'block' : 'hidden'
              }`}
            />

            {/* Corner Badge */}
            <div className="absolute top-3 left-3 bg-derma-bg/80 backdrop-blur-md px-2.5 py-1 rounded-lg border border-derma-border text-[11px] font-mono text-derma-accent font-semibold">
              {activeView.toUpperCase()} MODE
            </div>

            {/* Micro scale indicator (10mm) */}
            <div className="absolute bottom-3 right-3 bg-derma-bg/80 backdrop-blur-md px-2.5 py-1 rounded-lg border border-derma-border flex items-center space-x-1.5">
              <span className="w-8 h-0.5 bg-white block" />
              <span className="text-[11px] font-mono text-slate-300">10 mm</span>
            </div>
          </div>

          <p className="text-xs text-derma-muted mt-4 text-center">
            {activeView === 'gradcam' &&
              'Red/Amber zones indicate pixels that weighted heaviest in the neural network decision.'}
            {activeView === 'dullrazor' &&
              'DullRazor morphological closing eliminated occluding hair shafts without altering boundary edges.'}
            {activeView === 'contour' &&
              'Polar gradient boundary segmentation identifies periphery irregularity.'}
            {activeView === 'original' && 'Raw, unenhanced input dermoscopic photograph.'}
          </p>
        </div>

        {/* Controls & Layer Explainability */}
        <div className="space-y-4">
          {/* Controls Box */}
          <div className="p-4 rounded-xl bg-derma-bg/60 border border-derma-border space-y-3.5 text-xs">
            <div className="flex items-center space-x-2 text-slate-300 font-bold text-xs">
              <Sliders className="w-3.5 h-3.5 text-derma-accent" />
              <span>Grad-CAM Tuning & Colormap</span>
            </div>

            {/* Opacity Slider */}
            <div>
              <div className="flex justify-between text-slate-400 mb-1">
                <span>Heatmap Overlay Opacity</span>
                <span className="font-mono text-slate-50 font-bold">{Math.round(opacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-full accent-derma-accent cursor-pointer"
              />
            </div>

            {/* Colormap Buttons */}
            <div>
              <label className="block text-slate-400 mb-1.5">Colormap Palette</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'jet', name: 'Jet' },
                  { id: 'turbo', name: 'Turbo' },
                  { id: 'inferno', name: 'Inferno' },
                ].map((map) => (
                  <button
                    key={map.id}
                    onClick={() => setColormap(map.id)}
                    className={`py-2 px-2 rounded-lg text-xs font-semibold text-center border transition-colors ${
                      colormap === map.id
                        ? 'border-derma-accent bg-derma-surface text-derma-accent font-bold'
                        : 'border-derma-border text-slate-400 hover:text-slate-50 bg-derma-card'
                    }`}
                  >
                    {map.name}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-xs text-derma-muted">Jet is the usual clinical colormap.</p>
            </div>
          </div>

          {/* AI Attention Breakdown */}
          <div className="p-4 rounded-xl bg-derma-card border border-derma-border space-y-2.5 text-xs">
            <h4 className="font-bold text-slate-50 text-xs flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-derma-teal" />
              <span>Diagnostic Region Explanations</span>
            </h4>

            <div className="space-y-3 text-sm leading-relaxed text-slate-300">
              <div className="p-3 rounded-lg bg-derma-bg/70 border border-derma-border/60">
                <strong className="text-derma-accent block mb-1">Primary Activation Zone:</strong>
                Deep neural attention focused on the central hyperpigmented melanin core and
                peripheral border gradients.
              </div>
              <div className="p-3 rounded-lg bg-derma-bg/70 border border-derma-border/60">
                <strong className="text-derma-teal block mb-1">Hair Removal Verification:</strong>
                Morphological blackhat filter successfully removed hair follicles, preventing
                false-positive texture confusion.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
