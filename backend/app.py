"""
SkinScan AI — Production FastAPI Backend
Real-time multi-modal skin cancer triage with DenseNet121 + Grad-CAM + Doctor Referral.

Endpoints:
  POST /api/analyze  — Upload image + patient metadata → full triage report
  GET  /api/health   — Model status, GPU info, latency metrics
  GET  /api/benchmarks — Pre-loaded HAM10000 benchmark case metadata
  GET  /               — Service info
"""

import os
import time
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from inference import SkinScanInferenceEngine, get_engine

# ─────────────────────────────────────────────────────────
# APP CONFIGURATION
# ─────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = os.environ.get(
    "MODEL_WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "skinscan_densenet121_v2.pth"),
)
DEVICE = os.environ.get("DEVICE", "auto")  # "auto", "cpu", "cuda"
PORT = int(os.environ.get("BACKEND_PORT", 8000))

app = FastAPI(
    title="SkinScan AI Clinical Triage API",
    version="2.0.0",
    description="Production DenseNet121 Multi-Modal Skin Cancer Risk Stratification API",
)

# Enable CORS for React dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# STARTUP: Load Model
# ─────────────────────────────────────────────────────────
engine: Optional[SkinScanInferenceEngine] = None
last_inference_ms: float = 0


@app.on_event("startup")
async def startup_load_model():
    global engine
    device = None if DEVICE == "auto" else DEVICE
    engine = SkinScanInferenceEngine(device=device)

    if os.path.isfile(MODEL_WEIGHTS_PATH):
        try:
            engine.load_model(MODEL_WEIGHTS_PATH)
            print(f"[SkinScan API] Model loaded from: {MODEL_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[SkinScan API] WARNING: Failed to load model: {e}")
            print("[SkinScan API] Server running in DEMO mode (no real predictions)")
    else:
        print(f"[SkinScan API] No model weights found at: {MODEL_WEIGHTS_PATH}")
        print("[SkinScan API] Server running in DEMO mode.")
        print(
            "[SkinScan API] To enable real inference, place trained weights at:"
        )
        print(f"  {os.path.abspath(MODEL_WEIGHTS_PATH)}")

    # Serve frontend build if available
    dist_path = os.path.join(os.path.dirname(__file__), "..", "dist")
    if os.path.isdir(dist_path):
        app.mount("/", StaticFiles(directory=dist_path, html=True), name="frontend")
        print(f"[SkinScan API] Serving frontend from: {dist_path}")


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────
@app.get("/api")
def root():
    return {
        "status": "online",
        "service": "SkinScan AI Clinical Decision Support Engine",
        "version": "2.0.0",
        "model_backbone": "DenseNet121 Multi-Modal Hybrid + Focal Loss",
        "model_loaded": engine.model_loaded if engine else False,
    }


@app.get("/api/health")
def health():
    import torch

    return {
        "status": "healthy",
        "model_loaded": engine.model_loaded if engine else False,
        "model_version": "DenseNet121-MultiModal-FocalLoss-v2.0",
        "device": str(engine.device) if engine else "unknown",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "last_inference_ms": round(last_inference_ms, 1),
        "weights_path": MODEL_WEIGHTS_PATH,
        "weights_exist": os.path.isfile(MODEL_WEIGHTS_PATH),
    }


@app.post("/api/analyze")
async def analyze_lesion(
    age: int = Form(50),
    sex: str = Form("male"),
    localization: str = Form("back"),
    duration_months: int = Form(6),
    history_of_melanoma: bool = Form(False),
    file: Optional[UploadFile] = File(None),
):
    """
    Full multi-modal inference pipeline:
    1. DullRazor hair removal & color constancy
    2. DenseNet121 visual feature extraction (1024-d)
    3. Patient metadata MLP fusion (16-d)
    4. 7-class Focal Loss classification
    5. Grad-CAM explainability heatmap
    6. 4-tier risk stratification & DermaSensor 1-10 scoring
    7. Specialist doctor referral routing
    """
    global last_inference_ms

    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    if not engine.model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model weights not loaded",
                "message": "Place skinscan_densenet121_v2.pth in the models/ directory and restart the server.",
                "weights_path": os.path.abspath(MODEL_WEIGHTS_PATH),
            },
        )

    if file is None:
        raise HTTPException(status_code=400, detail="No image file uploaded")

    # Read image bytes
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    # Run inference
    try:
        start_time = time.perf_counter()

        result = engine.predict(
            image_bytes=image_bytes,
            age=age,
            sex=sex,
            localization=localization,
            duration_months=duration_months,
            history_of_melanoma=history_of_melanoma,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        last_inference_ms = elapsed_ms
        result["inference_time_ms"] = round(elapsed_ms, 1)

        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Inference pipeline error: {str(e)}"
        )


@app.get("/api/benchmarks")
def get_benchmarks():
    """Return metadata for the 7 pre-loaded HAM10000 benchmark cases."""
    return {
        "benchmarks": [
            {
                "code": "mel",
                "name": "Malignant Melanoma",
                "risk_stage": 4,
                "spectral_score": 9.2,
            },
            {
                "code": "bcc",
                "name": "Basal Cell Carcinoma",
                "risk_stage": 4,
                "spectral_score": 8.5,
            },
            {
                "code": "akiec",
                "name": "Actinic Keratosis",
                "risk_stage": 3,
                "spectral_score": 6.8,
            },
            {
                "code": "bkl",
                "name": "Benign Keratosis",
                "risk_stage": 2,
                "spectral_score": 4.2,
            },
            {
                "code": "nv",
                "name": "Melanocytic Nevus",
                "risk_stage": 1,
                "spectral_score": 1.8,
            },
            {
                "code": "vasc",
                "name": "Vascular Lesion",
                "risk_stage": 1,
                "spectral_score": 1.3,
            },
            {
                "code": "df",
                "name": "Dermatofibroma",
                "risk_stage": 1,
                "spectral_score": 2.0,
            },
        ]
    }


# ─────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SkinScan AI — Production Inference Server")
    print("  DenseNet121 Multi-Modal + Focal Loss + Grad-CAM")
    print(f"  Weights: {MODEL_WEIGHTS_PATH}")
    print(f"  Port: {PORT}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
