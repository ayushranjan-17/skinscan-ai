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
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from backend.inference import SkinScanInferenceEngine, get_engine
except ImportError:
    from inference import SkinScanInferenceEngine, get_engine

try:
    from backend.database import (
        init_db,
        create_user,
        authenticate_user,
        get_user_by_id,
        create_access_token,
        verify_access_token,
        create_scan,
        get_user_scans,
        get_scan_by_id,
        delete_user_scan,
        update_scan_feedback,
        get_retrain_queue,
        export_retrain_manifest,
        VALID_BIOPSY_LABELS,
    )
except ImportError:
    from database import (
        init_db,
        create_user,
        authenticate_user,
        get_user_by_id,
        create_access_token,
        verify_access_token,
        create_scan,
        get_user_scans,
        get_scan_by_id,
        delete_user_scan,
        update_scan_feedback,
        get_retrain_queue,
        export_retrain_manifest,
        VALID_BIOPSY_LABELS,
    )

# ─────────────────────────────────────────────────────────
# APP CONFIGURATION
# ─────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = os.environ.get(
    "MODEL_WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "skinscan_densenet121_v4.pth"),
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


def init_engine() -> SkinScanInferenceEngine:
    global engine
    if engine is None:
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
            print("[SkinScan API] To enable real inference, place trained weights at:")
            print(f"  {os.path.abspath(MODEL_WEIGHTS_PATH)}")
    return engine


@app.on_event("startup")
async def startup_load_model():
    init_db()
    init_engine()

    # Serve frontend build if available
    dist_path = os.path.join(os.path.dirname(__file__), "..", "dist")
    if os.path.isdir(dist_path):
        app.mount("/", StaticFiles(directory=dist_path, html=True), name="frontend")
        print(f"[SkinScan API] Serving frontend from: {dist_path}")


# ─────────────────────────────────────────────────────────
# AUTH SCHEMAS & DEPENDENCIES
# ─────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    role: str = Field(default="clinician")


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    confirmed_biopsy_label: str = Field(...)
    clinician_notes: Optional[str] = Field(default="")


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Extract authenticated user if valid Bearer token provided, else None (Guest Mode)."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:].strip()
    payload = verify_access_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Enforce authentication for protected endpoints."""
    user = get_current_user_optional(authorization)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ─────────────────────────────────────────────────────────
# AUTH & SCAN HISTORY ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/auth/register")
def register(req: RegisterRequest):
    """Register a new clinician or patient user."""
    try:
        user = create_user(
            name=req.name,
            email=req.email,
            password=req.password,
            role=req.role,
        )
        token = create_access_token(user["id"], user["email"], user["role"])
        return {"token": token, "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/auth/login")
def login(req: LoginRequest):
    """Authenticate clinician or patient credentials."""
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": user}


@app.get("/auth/me")
def get_me(user: dict = Depends(get_current_user)):
    """Return currently authenticated user profile."""
    return {"user": user}


@app.get("/scans")
def list_scans(user: dict = Depends(get_current_user)):
    """Return all scans recorded by the authenticated clinician/user."""
    scans = get_user_scans(user["id"])
    return {"scans": scans, "count": len(scans)}


@app.delete("/scans/{scan_id}")
def delete_scan(scan_id: str, user: dict = Depends(get_current_user)):
    """Delete a scan owned by the authenticated user."""
    success = delete_user_scan(user["id"], scan_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Scan '{scan_id}' not found or unauthorized")
    return {"status": "deleted", "scan_id": scan_id}


# ─────────────────────────────────────────────────────────
# HITL BIOPSY FEEDBACK & MLOPS QUEUE ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.patch("/scans/{scan_id}/feedback")
def submit_biopsy_feedback(
    scan_id: str,
    req: FeedbackRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Attach histopathology ground truth biopsy label and queue scan for offline model fine-tuning.
    """
    label = req.confirmed_biopsy_label.strip().lower()
    if label not in VALID_BIOPSY_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid biopsy label '{label}'. Must be one of: {sorted(list(VALID_BIOPSY_LABELS))}",
        )

    user = get_current_user_optional(authorization)
    updated = update_scan_feedback(
        scan_id=scan_id,
        confirmed_biopsy_label=label,
        clinician_notes=req.clinician_notes,
        user_id=user["id"] if user else None,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Scan '{scan_id}' not found")
    return {
        "status": "success",
        "message": "Biopsy ground truth recorded and queued for offline retraining",
        "scan": updated,
    }


@app.get("/mlops/retrain-queue")
def get_mlops_retrain_queue():
    """Return summary and cases queued for offline v5 retraining."""
    return get_retrain_queue()


@app.get("/mlops/export-queue")
def export_mlops_retrain_queue():
    """Export anonymized JSON manifest of verified biopsy cases for offline fine-tuning."""
    manifest = export_retrain_manifest()
    return {
        "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": "v4 (frozen)",
        "total_records": len(manifest),
        "manifest": manifest,
    }


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
    eng = init_engine()

    return {
        "status": "healthy",
        "model_loaded": eng.model_loaded if eng else False,
        "model_version": "DenseNet121-MultiModal-FocalLoss-v2.0",
        "device": str(eng.device) if eng else "unknown",
        "gpu_available": torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else ("Apple MPS" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else None)
        ),
        "last_inference_ms": round(last_inference_ms, 1),
        "weights_file": os.path.basename(MODEL_WEIGHTS_PATH),
        "weights_exist": os.path.isfile(MODEL_WEIGHTS_PATH),
    }


VALID_SEX_VALUES = {"male", "female", "other"}
VALID_LOCALIZATIONS = {
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
    "scalp",
    "neck",
    "unknown",
}
VALID_DOCTOR_PREFS = {"auto", "pcp", "derm", "onco", "mohs"}
VALID_LESION_TYPES = {"mole", "wound"}


@app.post("/api/analyze")
async def analyze_lesion(
    age: int = Form(50),
    sex: str = Form("male"),
    localization: str = Form("back"),
    duration_months: int = Form(6),
    history_of_melanoma: str = Form("false"),
    doctor_preference: str = Form("auto"),
    lesion_type: str = Form("mole"),
    file: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(None),
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
    eng = init_engine()

    if not eng:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    if not eng.model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model weights not loaded",
                "message": "Place skinscan_densenet121_v4.pth in the models/ directory and restart the server.",
            },
        )

    if file is None:
        raise HTTPException(status_code=400, detail="No image file uploaded")

    # Validate numeric ranges
    if age < 0 or age > 120:
        raise HTTPException(
            status_code=422,
            detail="Invalid patient age: must be between 0 and 120 years.",
        )
    if duration_months < 0 or duration_months > 1200:
        raise HTTPException(
            status_code=422,
            detail="Invalid lesion duration: must be between 0 and 1200 months.",
        )

    # Sanitize string enum inputs against allowlists
    clean_sex = sex.strip().lower()
    if clean_sex not in VALID_SEX_VALUES:
        clean_sex = "other"

    clean_loc = localization.strip().lower()
    if clean_loc not in VALID_LOCALIZATIONS:
        clean_loc = "unknown"

    clean_pref = doctor_preference.strip().lower()
    if clean_pref not in VALID_DOCTOR_PREFS:
        clean_pref = "auto"

    clean_lesion_type = lesion_type.strip().lower()
    if clean_lesion_type not in VALID_LESION_TYPES:
        clean_lesion_type = "mole"

    # Read and validate image bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image file exceeds 20MB limit")

    try:
        import io
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img_check:
            img_check.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file. Please upload a valid JPG, PNG, or WebP image.",
        )

    # Robust boolean parsing for history_of_melanoma
    melanoma_hx = history_of_melanoma.strip().lower() in ("true", "1", "yes", "t")

    # Run inference
    try:
        start_time = time.perf_counter()

        result = eng.predict(
            image_bytes=image_bytes,
            age=age,
            sex=clean_sex,
            localization=clean_loc,
            duration_months=duration_months,
            history_of_melanoma=melanoma_hx,
            doctor_preference=clean_pref,
            lesion_type=clean_lesion_type,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        last_inference_ms = elapsed_ms
        result["inference_time_ms"] = round(elapsed_ms, 1)

        # Expose top-level verification fields and aliases
        if "hair_removed_image" not in result:
            result["hair_removed_image"] = result.get("images", {}).get("dullrazor", "")
        if "hair_mask_image" not in result:
            result["hair_mask_image"] = result.get("images", {}).get("hair_mask", "")
        if "gradcam_image" not in result:
            result["gradcam_image"] = result.get("images", {}).get("gradcam", "")

        result["risk_stage"] = result.get("riskStage")
        result["risk_tier"] = result.get("cancer_verification", {}).get("risk_tier", result.get("riskLevel"))
        result["spectral_score"] = result.get("spectralScore")
        result["malignant_probability"] = result.get("cancer_verification", {}).get("malignant_probability")
        result["action_needed"] = result.get("actionNeeded")

        # Check authentication for persistent scan history (Guest Mode fallback)
        current_user = get_current_user_optional(authorization)
        if current_user:
            try:
                scan_id = create_scan(current_user["id"], result)
                result["scan_id"] = scan_id
                result["is_persisted"] = True
            except Exception as db_err:
                print(f"[SkinScan API] WARNING: Failed to persist scan: {db_err}")
                result["scan_id"] = None
                result["is_persisted"] = False
        else:
            result["scan_id"] = None
            result["is_persisted"] = False

        return JSONResponse(content=result)

    except HTTPException:
        raise
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
