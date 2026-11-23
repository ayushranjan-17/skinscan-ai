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
import logging
from contextlib import asynccontextmanager

logging.basicConfig(
    level=os.environ.get("SKINSCAN_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("skinscan.api")
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_pipeline.risk_stratification import BIOPSY_ALERT_THRESHOLD

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
        revoke_token,
        record_auth_attempt,
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
        revoke_token,
        record_auth_attempt,
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

# CORS allowlist (H1). Configure via SKINSCAN_CORS_ORIGINS as a comma-separated list.
# Defaults to the Vite dev server only; "*" is explicitly rejected when credentials are on.
_default_cors = "http://localhost:5173,http://127.0.0.1:5173"
_cors_raw = os.environ.get("SKINSCAN_CORS_ORIGINS", _default_cors)
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if "*" in CORS_ORIGINS:
    raise RuntimeError(
        "SKINSCAN_CORS_ORIGINS must be an explicit origin list; '*' is unsafe with credentialed requests."
    )

# Rate-limit thresholds (H4).
LOGIN_IP_LIMIT_PER_15MIN = int(os.environ.get("SKINSCAN_LOGIN_IP_LIMIT", "20"))
LOGIN_EMAIL_LIMIT_PER_15MIN = int(os.environ.get("SKINSCAN_LOGIN_EMAIL_LIMIT", "8"))
REGISTER_IP_LIMIT_PER_15MIN = int(os.environ.get("SKINSCAN_REGISTER_IP_LIMIT", "10"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + model; (nothing to tear down on shutdown)."""
    init_db()
    init_engine()
    dist_path = os.path.join(os.path.dirname(__file__), "..", "dist")
    if os.path.isdir(dist_path):
        app.mount("/app", StaticFiles(directory=dist_path, html=True), name="frontend")
        log.info("Serving frontend from %s at /app", dist_path)
    yield


app = FastAPI(
    title="SkinScan AI Clinical Triage API",
    version="2.0.0",
    description="Production DenseNet121 Multi-Modal Skin Cancer Risk Stratification API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def _client_key(request: Request) -> str:
    """Return a stable per-client identifier for rate-limit bucketing."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

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
                log.info("Model loaded from: %s", MODEL_WEIGHTS_PATH)
            except Exception as e:
                log.warning("Failed to load model (%s); running in DEMO mode", e)
        else:
            log.warning(
                "No model weights at %s; running in DEMO mode. Place weights at %s to enable inference.",
                MODEL_WEIGHTS_PATH,
                os.path.abspath(MODEL_WEIGHTS_PATH),
            )
    return engine


# ─────────────────────────────────────────────────────────
# AUTH SCHEMAS & DEPENDENCIES
# ─────────────────────────────────────────────────────────

# Roles anyone may pick at public sign-up. "researcher" is excluded on purpose (see register()).
PUBLIC_SIGNUP_ROLES = {"clinician", "patient"}


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=10)
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


def require_roles(*allowed: str):
    """Dependency factory: authenticated user whose role is in `allowed` (H2)."""
    allowed_set = set(allowed)

    def _dep(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in allowed_set:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user.get('role')}' is not authorized for this resource.",
            )
        return user

    return _dep


def _get_token_payload(authorization: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract the verified token payload (incl. jti/exp) without resolving the user."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return verify_access_token(authorization[7:].strip())


# ─────────────────────────────────────────────────────────
# AUTH & SCAN HISTORY ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/auth/register")
def register(req: RegisterRequest, request: Request):
    """Register a new clinician or patient user.

    Researchers can read the retraining queue and export, so that role is never self-assigned here; create
    researcher accounts on the server with `python3 -m backend.create_researcher`.
    """
    role = (req.role or "clinician").strip().lower()
    if role not in PUBLIC_SIGNUP_ROLES:
        raise HTTPException(
            status_code=403,
            detail="Only clinician or patient accounts can be created here. Ask an administrator for researcher access.",
        )
    ip = _client_key(request)
    attempts = record_auth_attempt(f"register:ip:{ip}")
    if attempts > REGISTER_IP_LIMIT_PER_15MIN:
        raise HTTPException(
            status_code=429,
            detail="Too many registrations from this address. Try again later.",
        )
    try:
        user = create_user(
            name=req.name,
            email=req.email,
            password=req.password,
            role=role,
        )
        token = create_access_token(user["id"], user["email"], user["role"])
        return {"token": token, "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/auth/login")
def login(req: LoginRequest, request: Request):
    """Authenticate clinician or patient credentials (rate-limited per IP and per email)."""
    ip = _client_key(request)
    email_key = req.email.strip().lower()
    ip_attempts = record_auth_attempt(f"login:ip:{ip}")
    email_attempts = record_auth_attempt(f"login:email:{email_key}")
    if ip_attempts > LOGIN_IP_LIMIT_PER_15MIN or email_attempts > LOGIN_EMAIL_LIMIT_PER_15MIN:
        raise HTTPException(
            status_code=429,
            detail="Too many sign-in attempts. Please wait a few minutes and try again.",
        )

    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": user}


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    """Revoke the caller's current token (H6)."""
    payload = _get_token_payload(authorization)
    if not payload:
        # Treat as idempotent: no-op if the token is missing or already invalid.
        return {"status": "ok"}
    jti = payload.get("jti")
    exp = int(payload.get("exp", 0))
    if jti:
        revoke_token(jti, exp)
    return {"status": "ok"}


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
    user: dict = Depends(require_roles("clinician", "researcher")),
):
    """
    Attach histopathology ground truth biopsy label and queue scan for offline model fine-tuning.
    Requires an authenticated clinician/researcher. Non-researchers may only label their own scans (H3).
    """
    label = req.confirmed_biopsy_label.strip().lower()
    if label not in VALID_BIOPSY_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid biopsy label '{label}'. Must be one of: {sorted(list(VALID_BIOPSY_LABELS))}",
        )

    # Researchers may annotate any scan; clinicians are scoped to their own scans.
    scope_user_id = None if user.get("role") == "researcher" else user["id"]
    updated = update_scan_feedback(
        scan_id=scan_id,
        confirmed_biopsy_label=label,
        clinician_notes=req.clinician_notes,
        user_id=scope_user_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Scan '{scan_id}' not found or not accessible")
    return {
        "status": "success",
        "message": "Biopsy ground truth recorded and queued for offline retraining",
        "scan": updated,
    }


@app.get("/mlops/retrain-queue")
def get_mlops_retrain_queue(user: dict = Depends(require_roles("researcher"))):
    """Return summary and cases queued for offline v5 retraining. Researcher-only (H2)."""
    return get_retrain_queue()


@app.get("/mlops/export-queue")
def export_mlops_retrain_queue(user: dict = Depends(require_roles("researcher"))):
    """Export anonymized JSON manifest of verified biopsy cases. Researcher-only (H2)."""
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
        "model_version": "DenseNet121-MultiModal-FocalLoss-v4",
        "alert_threshold": BIOPSY_ALERT_THRESHOLD,
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
                log.warning("Failed to persist scan: %s", db_err)
                result["scan_id"] = None
                result["is_persisted"] = False
        else:
            result["scan_id"] = None
            result["is_persisted"] = False

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Inference pipeline error")
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
    log.info("SkinScan AI — Production Inference Server (DenseNet121 Multi-Modal + Focal Loss + Grad-CAM)")
    log.info("Weights: %s", MODEL_WEIGHTS_PATH)
    log.info("Port: %s", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
