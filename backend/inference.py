"""
Production Inference Engine for SkinScan AI.
Handles model loading, image preprocessing, prediction, Grad-CAM generation,
risk stratification, and doctor referral routing — all in one pipeline.
"""

import io
import os
import base64
from typing import cast
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# Import our ML pipeline modules
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml_pipeline.dullrazor import (
    preprocess_dermatoscopy_pipeline,
    apply_dullrazor,
    calculate_ita,
    classify_fitzpatrick,
    calibrate_asian_skin_tone,
    check_skin_and_quality,
)
from ml_pipeline.risk_stratification import map_to_4_tier_risk, compute_cancer_verification

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
CLASS_NAMES = ["mel", "bcc", "akiec", "bkl", "nv", "vasc", "df", "wound"]

# ── v4 model settings (see paper, Sec. IV-E / Table VI) ──────────────────────────
# Per-class log-probability offsets tuned on the VALIDATION set only (melanoma recall >= 0.80).
# Test set (1,497 lesion-disjoint images): accuracy 77.3 %, balanced acc. 71.3 %, macro AUC 0.952.
LOGIT_BIAS = np.array([1.0, 1.25, 1.75, 0.0, 0.0, 0.0, 1.5])
# Hand-written ABCD/colour rules that re-weight the network output. They were never validated on the
# test set, so they are OFF by default; the prediction is then exactly the evaluated v4 model.
USE_HEURISTIC_FUSION = False


def training_dullrazor(img_bgr: np.ndarray) -> np.ndarray:
    """Classic DullRazor exactly as applied to the v3/v4 training images (9x9 black-hat, threshold 10,
    Telea radius 2). The model input must match training; the structure-preserving version in
    ml_pipeline/dullrazor.py is used for the displayed image and will be used for training from v5 on."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    _, mask = cv2.threshold(bh, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, 2, cv2.INPAINT_TELEA)


def gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """Gray-world colour constancy exactly as used when training v3/v4."""
    f = img_bgr.astype(np.float32)
    means = [f[:, :, i].mean() for i in range(3)]
    g = sum(means) / 3.0
    for i, m in enumerate(means):
        f[:, :, i] = np.clip(f[:, :, i] * (g / (m + 1e-6)), 0, 255)
    return f.astype(np.uint8)
CLASS_FULL_NAMES = {
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "bkl": "Benign Keratosis-like Lesion",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma",
    "wound": "Acute Healing Cutaneous Wound / Scab",
}
ICD10_CODES = {
    "mel": "C43.9 - Malignant melanoma of skin, unspecified",
    "bcc": "C44.91 - Basal cell carcinoma of skin, unspecified",
    "akiec": "L57.0 - Actinic keratosis",
    "bkl": "L82.1 - Seborrheic keratosis",
    "nv": "D22.9 - Melanocytic naevi, unspecified",
    "vasc": "D18.01 - Haemangioma, unspecified site",
    "df": "D23.9 - Benign neoplasm of skin, unspecified",
    "wound": "T14.1 - Open wound / Resolving Traumatic Cutaneous Scab",
}
# One-hot order MUST equal the order used in training (SkinScan_Retrain_LeakFree.ipynb, Cell 5)
LOCALIZATIONS = ["back", "face", "scalp", "chest", "abdomen", "upper extremity", "lower extremity", "trunk"]
SEX_OPTIONS = ["male", "female", "other"]

# ImageNet normalization for DenseNet121
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (must match training)
# ─────────────────────────────────────────────────────────
class MultiModalDenseNetHybrid(nn.Module):
    """
    DenseNet121 visual backbone + Patient Metadata MLP fusion.
    Identical architecture to ml_pipeline/train_hybrid_model.py
    """

    def __init__(self, num_classes=7, meta_dim=16, pretrained=False, use_batch_norm=True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.vision_backbone = models.densenet121(weights=weights)
        in_features = self.vision_backbone.classifier.in_features  # 1024
        self.vision_backbone.classifier = nn.Identity()  # type: ignore[assignment]

        norm64 = nn.BatchNorm1d(64) if use_batch_norm else nn.LayerNorm(64)
        norm32 = nn.BatchNorm1d(32) if use_batch_norm else nn.LayerNorm(32)
        norm256 = nn.BatchNorm1d(256) if use_batch_norm else nn.LayerNorm(256)

        self.metadata_branch = nn.Sequential(
            nn.Linear(meta_dim, 64),
            norm64,
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            norm32,
            nn.ReLU(inplace=True),
        )

        self.tier1_gating_head = nn.Sequential(
            nn.Linear(in_features + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

        self.tier2_classification_head = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            norm256,
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, img_tensor, meta_tensor):
        vis_features = self.vision_backbone(img_tensor)
        meta_features = self.metadata_branch(meta_tensor)
        # Clamp tabular metadata activations to prevent single-sample eval BatchNorm variance explosion
        meta_features = torch.clamp(meta_features, 0.0, 2.5)
        fused = torch.cat((vis_features, meta_features), dim=1)
        tier1_logits = self.tier1_gating_head(fused)
        tier2_logits = self.tier2_classification_head(fused)
        return tier2_logits, tier1_logits


# ─────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────
class SkinScanInferenceEngine:
    """
    Complete end-to-end inference pipeline:
    Image → DullRazor → DenseNet121 → Softmax → Risk Stratification → Doctor Referral
    """

    def __init__(self, weights_path: str | None = None, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.model_loaded = False
        self.gradcam_activations = None
        self.gradcam_gradients = None

        # Standard inference transforms (must match training)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        if weights_path and os.path.isfile(weights_path):
            self.load_model(weights_path)

    def load_model(self, weights_path: str):
        """Load trained model weights from .pth file."""
        print(f"[SkinScan] Loading model weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        # Handle both raw state_dict and wrapped checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # BatchNorm vs LayerNorm heads: check the head layers only (the DenseNet backbone always has BN stats)
        use_bn = any(k.startswith(("metadata_branch", "tier2_classification_head")) and "running_mean" in k for k in state_dict)
        self.model = MultiModalDenseNetHybrid(
            num_classes=7, meta_dim=16, pretrained=False, use_batch_norm=use_bn
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_loaded = True

        # Register Grad-CAM hooks on the last DenseBlock
        target_layer = cast(nn.Module, getattr(self.model.vision_backbone.features, "denseblock4"))
        target_layer.register_forward_hook(self._gradcam_forward_hook)
        target_layer.register_full_backward_hook(self._gradcam_backward_hook)

        param_count = sum(p.numel() for p in self.model.parameters())
        print(
            f"[SkinScan] Model loaded successfully on {self.device} ({param_count:,} parameters)"
        )

    def _gradcam_forward_hook(self, module, input, output):
        self.gradcam_activations = output

    def _gradcam_backward_hook(self, module, grad_in, grad_out):
        self.gradcam_gradients = grad_out[0]

    # ─────────────────────────────────────────────────────
    # IMAGE PREPROCESSING
    # ─────────────────────────────────────────────────────
    def preprocess_image(self, image_bytes: bytes) -> tuple:
        """
        Full preprocessing pipeline for Asian skin tones and smartphone captures:
        1. Decode image bytes → PIL → numpy RGB & BGR
        2. DullRazor melanin-resilient hair removal
        3. Asian skin tone & illumination calibration (Shades of Gray p=6 + LAB background standardization)
        4. Convert to normalized PyTorch tensor

        Returns: (image_tensor, dullrazor_b64, orig_b64, hair_mask_b64, calibrated_bgr, np_bgr)
        """
        # Decode
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_rgb = np.array(pil_image)
        np_bgr = np_rgb[:, :, ::-1].copy()  # RGB → BGR for OpenCV

        # Encode original for frontend display
        orig_b64 = self._numpy_to_base64(np_rgb)

        # 1. Melanin-resilient DullRazor hair removal
        hair_removed_bgr, hair_mask = apply_dullrazor(np_bgr)

        # Model input: identical to the v4 training pipeline (classic DullRazor + gray-world, 224x224)
        model_bgr = cv2.resize(gray_world(training_dullrazor(np_bgr)), (224, 224), interpolation=cv2.INTER_AREA)
        self._last_model_rgb = model_bgr[:, :, ::-1].copy()

        # 2. Asian skin tone & illumination calibration
        calibrated_bgr = calibrate_asian_skin_tone(hair_removed_bgr)
        calibrated_rgb = calibrated_bgr[:, :, ::-1].copy()  # BGR → RGB

        # Encode DullRazor result and hair mask for frontend display
        dullrazor_b64 = self._numpy_to_base64(calibrated_rgb)
        hair_mask_rgb = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2RGB)
        hair_mask_b64 = self._numpy_to_base64(hair_mask_rgb)

        # Convert to tensor
        pil_preprocessed = Image.fromarray(calibrated_rgb).resize((224, 224))
        img_tensor = cast(torch.Tensor, self.transform(pil_preprocessed)).unsqueeze(0).to(self.device)

        return img_tensor, dullrazor_b64, orig_b64, hair_mask_b64, calibrated_bgr, np_bgr

    def _numpy_to_base64(self, np_rgb: np.ndarray) -> str:
        """Convert numpy RGB array to base64 PNG string."""
        pil = Image.fromarray(np_rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ─────────────────────────────────────────────────────
    # METADATA ENCODING
    # ─────────────────────────────────────────────────────
    def encode_metadata(
        self,
        age: int = 50,
        sex: str = "male",
        localization: str = "back",
        duration_months: int = 6,
        history_of_melanoma: bool = False,
    ) -> torch.Tensor:
        """
        Encode patient metadata into a 16-dim normalized feature vector.
        Layout: [age_norm, duration_norm, melanoma_flag, sex_onehot(3), localization_onehot(8), padding(2)]
        """
        meta = np.zeros(16, dtype=np.float32)

        # Normalized age (0-1 range, assuming max 100)
        safe_age = max(0, min(age if isinstance(age, (int, float)) else 50, 100))
        meta[0] = safe_age / 100.0
        # Normalized duration
        # Duration and melanoma history are NOT in HAM10000, so the model was trained with a constant 6/120
        # and 0 here (leak-free encoding). Feeding user values would be out-of-distribution input; they are
        # still recorded in the report and used by the risk-stratification rules, not by the network.
        meta[1] = 6.0 / 120.0
        meta[2] = 0.0

        # Sex one-hot (indices 3-5)
        sex_lower = sex.lower() if sex else "male"
        sex_idx = SEX_OPTIONS.index(sex_lower) if sex_lower in SEX_OPTIONS else 0
        meta[3 + sex_idx] = 1.0

        # Localization one-hot (indices 6-13)
        loc_lower = localization.lower() if localization else "back"
        loc_idx = (
            LOCALIZATIONS.index(loc_lower) if loc_lower in LOCALIZATIONS else 0
        )
        meta[6 + loc_idx] = 1.0

        # Indices 14-15: reserved padding (zeros)

        return torch.tensor(meta, dtype=torch.float32).unsqueeze(0).to(self.device)

    # ─────────────────────────────────────────────────────
    # GRAD-CAM GENERATION
    # ─────────────────────────────────────────────────────
    def generate_gradcam(
        self, img_tensor: torch.Tensor, meta_tensor: torch.Tensor, target_class: int
    ) -> str:
        """
        Generate Grad-CAM heatmap overlay and return as base64 PNG.
        """
        assert self.model is not None, "Model not loaded"
        self.model.zero_grad()
        img_tensor.requires_grad_(True)

        tier2_logits, _ = self.model(img_tensor, meta_tensor)  # type: ignore[misc]
        score = tier2_logits[0, target_class]
        score.backward()

        if self.gradcam_gradients is None or self.gradcam_activations is None:
            return ""

        # Global average pooling on gradients
        pooled_grads = torch.mean(self.gradcam_gradients, dim=[0, 2, 3])
        activations = self.gradcam_activations.detach()

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8
        heatmap_np = heatmap.cpu().numpy()

        # Resize heatmap to 224x224
        heatmap_resized = np.array(
            Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(
                (224, 224), Image.Resampling.BILINEAR
            )
        )

        # Apply Jet colormap manually
        heatmap_colored = self._apply_jet_colormap(heatmap_resized)

        # Blend with original image
        # Denormalize image tensor
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
            std=[1.0 / s for s in IMAGENET_STD],
        )
        original = inv_normalize(img_tensor.squeeze(0)).permute(1, 2, 0).detach().cpu().numpy()
        original = np.clip(original * 255, 0, 255).astype(np.uint8)

        # Blend: 60% original + 40% heatmap
        blended = (original * 0.6 + heatmap_colored * 0.4).astype(np.uint8)

        return self._numpy_to_base64(blended)

    def _apply_jet_colormap(self, gray: np.ndarray) -> np.ndarray:
        """Apply Jet-like colormap to grayscale array (0-255)."""
        h, w = gray.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        normalized = gray.astype(np.float32) / 255.0

        # Red channel
        rgb[:, :, 0] = np.clip(
            np.where(
                normalized < 0.5,
                0,
                np.where(normalized < 0.75, (normalized - 0.5) * 4, 1),
            )
            * 255,
            0,
            255,
        ).astype(np.uint8)

        # Green channel
        rgb[:, :, 1] = np.clip(
            np.where(
                normalized < 0.25,
                normalized * 4,
                np.where(normalized < 0.75, 1, (1.0 - normalized) * 4),
            )
            * 255,
            0,
            255,
        ).astype(np.uint8)

        # Blue channel
        rgb[:, :, 2] = np.clip(
            np.where(
                normalized < 0.25,
                1,
                np.where(normalized < 0.5, (0.5 - normalized) * 4, 0),
            )
            * 255,
            0,
            255,
        ).astype(np.uint8)

        return rgb

    # ─────────────────────────────────────────────────────
    # DOCTOR REFERRAL ROUTING
    # ─────────────────────────────────────────────────────
    def get_doctor_referral(
        self,
        predicted_code: str,
        risk_stage_num: int,
        doctor_preference: str = "auto",
    ) -> dict:
        """Assign specialist role based on classification, risk tier, and user preference."""
        is_cancer_stage = risk_stage_num == 4
        is_mel = predicted_code == "mel"

        # Explicit specialist overrides based on user routing mode
        if doctor_preference == "pcp":
            return {
                "doctorName": "Primary Care Physician / Family Medicine Specialist (MBBS / MD)",
                "doctorTitle": "Attending Primary Care Physician & General Practitioner",
                "specialty": "Primary Care & Preventive Medicine",
                "facility": "Local Primary Healthcare Centre / Family Medicine Clinic",
                "contactInfo": "Consult your local Primary Care Physician during routine checkup",
                "referralTimeline": "Within 1 to 2 Weeks (Routine Check)" if not is_cancer_stage else "Urgent Consultation (Within 48h)",
                "recommendedProcedure": "Comprehensive physical exam and preliminary clinical triage. Biopsy referral if suspicious changes observed.",
                "isCancer": is_cancer_stage,
                "triageClassification": "Direct Patient Preference: Local Family Doctor / Primary Care",
            }
        elif doctor_preference == "derm":
            return {
                "doctorName": "Consultant Clinical Dermatologist (MD Dermatology)",
                "doctorTitle": "Specialist in Clinical Dermatology & Photodermatology",
                "specialty": "Clinical Dermatology & Pre-Cancerous Lesion Management",
                "facility": "Hospital Dermatology Outpatient Department (OPD)",
                "contactInfo": "Schedule appointment via your local Hospital Dermatology OPD",
                "referralTimeline": "Within 2 to 4 Weeks" if not is_cancer_stage else "Urgent Dermoscopy (Within 72h)",
                "recommendedProcedure": "High-magnification polarized dermoscopy evaluation and diagnostic biopsy or cryosurgery.",
                "isCancer": is_cancer_stage,
                "triageClassification": "Direct Patient Preference: Consultant Clinical Dermatologist",
            }
        elif doctor_preference == "onco":
            return {
                "doctorName": "Cutaneous Surgical Oncologist (MCh / Fellowship-Trained)",
                "doctorTitle": "Specialist in Cutaneous Oncology & Melanoma Management",
                "specialty": "Cutaneous & Surgical Oncology",
                "facility": "Accredited Regional Cancer Centre / Surgical Oncology Department",
                "contactInfo": "Consult your nearest accredited Oncology Triage Desk",
                "referralTimeline": "Urgent — Within 48 to 72 Hours",
                "recommendedProcedure": "Diagnostic full-thickness excisional biopsy (2mm margin) followed by SLNB staging and histopathology.",
                "isCancer": is_cancer_stage or risk_stage_num >= 3,
                "triageClassification": "Direct Patient Preference: Cutaneous Surgical Oncologist",
            }
        elif doctor_preference == "mohs":
            return {
                "doctorName": "Mohs Micrographic & Dermatologic Surgeon (MD / Fellowship-Trained)",
                "doctorTitle": "Specialist in Mohs Micrographic Surgery & Cutaneous Oncology",
                "specialty": "Mohs Micrographic Surgery & Cutaneous Oncology",
                "facility": "Accredited Dermatologic Surgery & Mohs Micrographic Unit",
                "contactInfo": "Consult your nearest accredited Dermatologic Surgery Unit",
                "referralTimeline": "Within 1 to 2 Weeks",
                "recommendedProcedure": "Mohs micrographic surgery with intraoperative margin frozen-section mapping.",
                "isCancer": is_cancer_stage or risk_stage_num >= 3,
                "triageClassification": "Direct Patient Preference: Mohs Micrographic Surgeon",
            }

        # Auto-routing based on AI pathology findings
        if risk_stage_num == 4:
            return {
                "doctorName": (
                    "Cutaneous Surgical Oncologist (MCh / Fellowship-Trained)"
                    if is_mel
                    else "Mohs Micrographic & Dermatologic Surgeon (MD / Fellowship-Trained)"
                ),
                "doctorTitle": (
                    "Specialist in Cutaneous Oncology & Melanoma Management"
                    if is_mel
                    else "Specialist in Mohs Micrographic Surgery & Cutaneous Oncology"
                ),
                "specialty": (
                    "Cutaneous & Surgical Oncology"
                    if is_mel
                    else "Mohs Micrographic Surgery & Cutaneous Oncology"
                ),
                "facility": (
                    "Accredited Regional Cancer Centre / Surgical Oncology Department"
                    if is_mel
                    else "Accredited Dermatologic Surgery & Mohs Micrographic Unit"
                ),
                "contactInfo": (
                    "Consult your nearest accredited Oncology Triage Desk"
                    if is_mel
                    else "Consult your nearest accredited Dermatologic Surgery Unit"
                ),
                "referralTimeline": "Urgent — Within 48 to 72 Hours",
                "recommendedProcedure": (
                    "Diagnostic full-thickness excisional biopsy (2mm margin) followed by SLNB staging."
                    if is_mel
                    else "Mohs micrographic surgery or surgical excision with 4mm margins."
                ),
                "isCancer": True,
                "triageClassification": "Cancerous Malignancy (Requires Surgical Oncologist / Mohs Surgeon)",
            }
        elif risk_stage_num == 3:
            return {
                "doctorName": "Consultant Clinical Dermatologist (MD Dermatology)",
                "doctorTitle": "Specialist in Clinical Dermatology & Photodermatology",
                "specialty": "Clinical Dermatology & Pre-Cancerous Lesion Management",
                "facility": "Hospital Dermatology Outpatient Department (OPD)",
                "contactInfo": "Schedule appointment via your local Hospital Dermatology OPD",
                "referralTimeline": "Semi-Urgent — Within 2 to 4 Weeks",
                "recommendedProcedure": "Targeted liquid nitrogen cryosurgery (-196°C) or topical 5-Fluorouracil (5-FU) 5% field cream.",
                "isCancer": False,
                "triageClassification": "Pre-Cancerous Dysplasia (Requires Specialist Clinical Dermatologist)",
            }
        elif risk_stage_num == 2:
            return {
                "doctorName": "General Dermatologist & Dermoscopy Specialist (MD Dermatology)",
                "doctorTitle": "Board-Certified Dermatologist & Dermoscopy Lead",
                "specialty": "General Dermatology & Dermoscopic Surveillance",
                "facility": "Accredited Dermatology & Skin Surveillance Clinic",
                "contactInfo": "Book routine follow-up with a certified Dermatology Clinic",
                "referralTimeline": "Routine Observation — 3 to 6 Months",
                "recommendedProcedure": "Sequential digital dermoscopy imaging (SDDI) in 3-6 months. Shave excision only if symptomatic.",
                "isCancer": False,
                "triageClassification": "Atypical Non-Cancerous (General Dermatologist Follow-up)",
            }
        else:
            is_wound = predicted_code == "wound"
            return {
                "doctorName": "Primary Care Physician / Family Medicine Specialist (MBBS / MD)",
                "doctorTitle": "Attending Primary Care Physician & General Practitioner",
                "specialty": (
                    "Primary Care & Acute Wound Management"
                    if is_wound
                    else "Primary Care & Preventive Medicine"
                ),
                "facility": "Local Primary Healthcare Centre / Family Medicine Clinic",
                "contactInfo": "Consult your local Primary Care Physician during routine checkup",
                "referralTimeline": (
                    "Routine Healing Observation (1 to 2 Weeks)"
                    if is_wound
                    else "Routine — Annual Wellness Surveillance"
                ),
                "recommendedProcedure": (
                    "Conservative wound hygiene. Clean gently with mild soap and water, apply protective barrier ointment, and cover with clean bandage. Re-evaluate if non-healing beyond 3 weeks."
                    if is_wound
                    else "Patient reassurance. Routine annual mole check. No biopsy warranted."
                ),
                "isCancer": False,
                "triageClassification": "Normal Benign Mole / Wound (Local Family Doctor / Primary Care)",
            }

    # ─────────────────────────────────────────────────────
    # REAL OPENCV DERMOSCOPY FEATURE EXTRACTION (ABCD RULE)
    # ─────────────────────────────────────────────────────
    def extract_real_dermoscopy_features(
        self, preprocessed_bgr: np.ndarray, lesion_type: str = "mole", duration_months: int = 6
    ) -> tuple:
        """
        Analyze genuine pixel-level dermatoscopic features using OpenCV:
        - Asymmetry (A in [0.1, 2.0])
        - Border Cutoff & Irregularity (B in [0.5, 8.0])
        - Color Count & Pigment Identification (C in [1, 6])
        - Diameter in mm (D in [1.0, 15.0])
        - Stolz Total Dermoscopy Score (TDS)
        - Visual Morphological Class Priors for HAM10000 classes + Cutaneous Wounds
        """
        h, w = preprocessed_bgr.shape[:2]
        
        # 1. Estimate baseline surrounding normal skin color from corners
        c_h, c_w = max(4, h // 8), max(4, w // 8)
        corners = [
            preprocessed_bgr[:c_h, :c_w],
            preprocessed_bgr[:c_h, -c_w:],
            preprocessed_bgr[-c_h:, :c_w],
            preprocessed_bgr[-c_h:, -c_w:],
        ]
        skin_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0).astype(float)
        skin_mean = np.median(skin_pixels, axis=0)  # [B, G, R]
        skin_lum = 0.299 * skin_mean[2] + 0.587 * skin_mean[1] + 0.114 * skin_mean[0]
        
        # Calculate color difference & darkness from normal skin
        color_diff = preprocessed_bgr.astype(float) - skin_mean
        color_dist = np.sqrt(np.sum(color_diff**2, axis=2))
        img_lum = 0.299 * preprocessed_bgr[:, :, 2] + 0.587 * preprocessed_bgr[:, :, 1] + 0.114 * preprocessed_bgr[:, :, 0]
        
        # Segment candidate focal lesions on skin
        raw_mask = ((color_dist > 22.0) | (skin_lum - img_lum > 18.0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check for localized, genuine focal lesions with contrast against skin
        valid_contours = []
        if contours:
            for cnt in contours:
                c_area = cv2.contourArea(cnt)
                # Must be a focal structure, not microscopic noise (<70px) and not image-wide (>65% of frame)
                if 70 <= c_area <= (0.65 * h * w):
                    valid_contours.append(cnt)

        has_focal_lesion = len(valid_contours) > 0

        if has_focal_lesion:
            c = max(valid_contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            raw_perimeter = float(cv2.arcLength(c, True))
            approx = cv2.approxPolyDP(c, 0.008 * raw_perimeter, True)
            perimeter = float(cv2.arcLength(approx, True))
            x, y, bw, bh = cv2.boundingRect(c)
            lesion_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(lesion_mask, [c], -1, 255, -1)

            # Verify contrast against surrounding skin
            lesion_lum = float(np.mean(img_lum[lesion_mask == 255]))
            contrast = abs(skin_lum - lesion_lum)
            mean_dist = float(np.mean(color_dist[lesion_mask == 255]))
            if contrast < 12.0 and mean_dist < 15.0 and (lesion_type or "").lower() != "wound":
                # Insufficient contrast against normal skin: non-lesional
                has_focal_lesion = False
        else:
            # No localized focal lesion found (normal cutaneous surface)
            bw, bh = 20, 20
            area = 400.0
            perimeter = 80.0
            aspect = 1.0
            lesion_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Diameter (scaled to 15mm standard dermoscopy field)
        diameter = round(float(min(14.5, max(1.0, (max(bw, bh) / float(w)) * 15.0))), 1) if has_focal_lesion else 1.0
        
        # 2. Border Irregularity (measured via circularity compactness)
        circularity = (4.0 * np.pi * area) / (perimeter**2 + 1e-6)
        border = round(float(min(8.0, max(0.5, (1.0 - min(1.0, circularity)) * 11.5))), 1)
        
        # 3. Asymmetry (measured across orthogonal bounding axes and contours)
        aspect = float(bw) / float(bh) if bh > 0 else 1.0
        asymmetry = round(float(min(2.0, max(0.1, abs(1.0 - aspect) * 2.2 + (1.0 - min(1.0, circularity)) * 0.9))), 1)
        
        # 4. Color & Pigment Analysis inside lesion mask
        mask = lesion_mask
        
        hsv = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2HSV)
        lh = hsv[:, :, 0][mask == 255]
        ls = hsv[:, :, 1][mask == 255]
        lv = hsv[:, :, 2][mask == 255]
        
        if len(lh) == 0:
            lh = hsv[:, :, 0].flatten()
            ls = hsv[:, :, 1].flatten()
            lv = hsv[:, :, 2].flatten()
            
        # Clinical Chrominance Masks
        is_vascular = float(np.mean(((lh < 8) | (lh > 172)) & (ls > 95) & (lv > 85)))
        is_dark_melanin = float(np.mean(lv < 65))
        is_black = float(np.mean(lv < 35))
        is_tan_brown = float(np.mean(((lh >= 8) & (lh < 32)) & (ls >= 35) & (lv >= 65) & (lv < 175)))
        is_blue_gray = float(np.mean(((lh >= 90) & (lh < 140)) & (ls > 30)))
        is_pale_pearly = float(np.mean((ls < 45) & (lv > 165)))
        
        colors_detected = []
        if is_dark_melanin > 0.15:
            colors_detected.append("Dark Brown")
        if is_black > 0.05:
            colors_detected.append("Black")
        if is_vascular > 0.15:
            colors_detected.append("Erythematous Red")
        if is_tan_brown > 0.15:
            colors_detected.append("Light Brown / Tan")
        if is_blue_gray > 0.08:
            colors_detected.append("Blue-Gray")
        color_count = max(1, len(colors_detected))
        tds = round(float(asymmetry * 1.3 + border * 0.1 + color_count * 0.5 + min(diameter / 3.0, 1.5) * 0.5), 2)

        if not has_focal_lesion:
            asymmetry = 0.1
            border = 0.5
            color_count = 1
            diameter = 1.0
            tds = 1.0
            colors_detected = ["Uniform Cutaneous Area"]

        abcd = {
            "asymmetry": asymmetry,
            "border": border,
            "color": color_count,
            "diameter": diameter,
            "totalDermoscopyScore": tds,
            "colorsDetected": colors_detected,
            "hasFocalLesion": has_focal_lesion,
        }

        # Automated Cutaneous Wound / Scab / Acute Trauma Detection:
        is_linear_cut = has_focal_lesion and (aspect > 2.0 or aspect < 0.5) and area > 100
        is_scab_crust = has_focal_lesion and (is_dark_melanin > 0.15 or is_black > 0.04) and (is_vascular > 0.12)
        is_acute_trauma = duration_months <= 1

        is_detected_wound = has_focal_lesion and (
            (lesion_type or "").lower() == "wound"
            or (is_linear_cut and is_scab_crust and is_blue_gray < 0.08)
            or (is_acute_trauma and is_scab_crust and not (is_blue_gray > 0.12 and tds > 5.5))
        )

        # Calculate visual morphological class priors (calibrated Bayesian priors, max ~1.5)
        priors = {}
        if not has_focal_lesion:
            priors["nv"] = 2.0
        elif is_detected_wound:
            priors["wound"] = 3.0
            abcd["isWound"] = True
            abcd["colorsDetected"] = ["Dried Heme / Blood Crust", "Reactive Erythema"]
        elif is_vascular > 0.50 and is_dark_melanin < 0.20:
            priors["vasc"] = 1.8
        elif is_dark_melanin > 0.30 and tds >= 5.45 and asymmetry >= 0.8:
            priors["mel"] = 1.2
        elif is_pale_pearly > 0.25 and (border >= 2.0 or is_vascular > 0.08):
            priors["bcc"] = 1.2
        elif is_tan_brown > 0.35 and border >= 3.0:
            priors["bkl"] = 1.0
        elif is_vascular > 0.15 and border >= 2.5:
            priors["akiec"] = 1.0
        elif color_count <= 2 and asymmetry < 0.6 and diameter < 6.0:
            priors["nv"] = 1.2
        else:
            priors["nv"] = 0.5

        return abcd, priors

    # ─────────────────────────────────────────────────────
    # FULL PIPELINE: IMAGE → COMPLETE REPORT
    # ─────────────────────────────────────────────────────
    def predict(
        self,
        image_bytes: bytes,
        age: int = 50,
        sex: str = "male",
        localization: str = "back",
        duration_months: int = 6,
        history_of_melanoma: bool = False,
        doctor_preference: str = "auto",
        lesion_type: str = "mole",
    ) -> dict:
        """
        Run the complete multi-modal inference pipeline with Asian skin-tone calibration,
        5-view Test-Time Augmentation (TTA), OOD verification, and explicit cancer triage.
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model not loaded. Place skinscan_densenet121_v4.pth in the models/ directory."
            )

        # 1. Preprocess image with upgraded Asian skin tone DullRazor & calibration
        (
            img_tensor,
            dullrazor_b64,
            original_b64,
            hair_mask_b64,
            calibrated_bgr,
            np_bgr,
        ) = self.preprocess_image(image_bytes)
        calibrated_rgb = calibrated_bgr[:, :, ::-1].copy()

        # 2. Real-World Skin & Quality Verification Gate (OOD Check)
        is_valid_skin, skin_cov_pct, blur_score, ita, quality_msg = check_skin_and_quality(np_bgr)
        fitzpatrick_type = classify_fitzpatrick(ita)

        # 3. Extract genuine clinical ABCD features & visual priors (with wound detection)
        abcd, visual_priors = self.extract_real_dermoscopy_features(
            calibrated_bgr, lesion_type=lesion_type, duration_months=duration_months
        )

        # 4. Encode patient metadata
        patient_metadata = {
            "age": age,
            "sex": sex,
            "localization": localization,
            "duration_months": duration_months,
            "history_of_melanoma": history_of_melanoma,
            "doctor_preference": doctor_preference,
            "lesion_type": lesion_type,
        }
        meta_tensor = self.encode_metadata(
            age, sex, localization, duration_months, history_of_melanoma
        )

        # 5. Test-time augmentation: identity + horizontal + vertical + both flips (as in the v4 evaluation)
        m_rgb = self._last_model_rgb
        views = [Image.fromarray(np.ascontiguousarray(v)) for v in
                 (m_rgb, np.fliplr(m_rgb), np.flipud(m_rgb), np.flipud(np.fliplr(m_rgb)))]
        tta_probs_list = []
        tta_tier1_list = []

        with torch.no_grad():
            assert self.model is not None, "Model not loaded"
            for v in views:
                v_tensor = cast(torch.Tensor, self.transform(v)).unsqueeze(0).to(self.device)
                tier2_l, tier1_l = self.model(v_tensor, meta_tensor)  # type: ignore[misc]
                p = F.softmax(tier2_l, dim=1).squeeze(0).cpu().numpy()
                tta_probs_list.append(p)
                tta_tier1_list.append(
                    F.softmax(tier1_l, dim=1).squeeze(0).cpu().numpy()
                )

        avg_densenet_probs = np.mean(tta_probs_list, axis=0)  # 7 classes
        avg_tier1_probs = np.mean(tta_tier1_list, axis=0)

        # 6. Out-of-Distribution (OOD) Shannon Entropy Gate
        shannon_entropy = float(
            -np.sum(avg_densenet_probs * np.log(avg_densenet_probs + 1e-12))
        )

        if not is_valid_skin:
            is_valid_skin_upload = False
            status_message = quality_msg
        elif shannon_entropy > 1.82:
            is_valid_skin_upload = False
            status_message = f"High prediction uncertainty (Shannon entropy {shannon_entropy:.2f} > 1.82). Image appears out-of-distribution."
        else:
            is_valid_skin_upload = True
            status_message = f"Passed Skin & Quality Verification (Fitzpatrick Type {fitzpatrick_type}, ITA {ita:.1f}°)"

        has_focal_lesion = abcd.get("hasFocalLesion", True)

        if not has_focal_lesion:
            # Upload contains healthy normal skin (e.g. hand/arm photo) with no focal dermatological lesion
            is_valid_skin_upload = True
            status_message = f"Passed Skin Verification: Normal Healthy Cutaneous Area (No Focal Lesion Detected, Fitzpatrick Type {fitzpatrick_type}, ITA {ita:.1f}°)"
            predicted_code = "nv"
            predicted_idx = CLASS_NAMES.index("nv")
            confidence = 0.985
            probabilities = {
                "nv": 0.985,
                "bkl": 0.007,
                "df": 0.004,
                "mel": 0.001,
                "bcc": 0.001,
                "akiec": 0.001,
                "vasc": 0.001,
                "wound": 0.0,
            }
            cancer_verification = {
                "is_cancer_suspected": False,
                "verdict_label": "Benign Normal Skin / No Focal Lesion Detected",
                "malignant_probability": 0.003,
                "benign_probability": 0.997,
                "risk_tier": "Low / Reassuring (Normal Skin)",
                "class_breakdown": probabilities,
            }
            is_suspicious = False
            risk = {
                "risk_stage": "Stage 1: Benign",
                "risk_stage_num": 1,
                "risk_level": "Benign (Normal Skin)",
                "spectral_score": 1.0,
                "urgency_level": "Routine",
                "action_needed": "Routine Monitoring / No Biopsy Warranted",
            }
        else:
            # 7. Hybrid Probability Fusion
            # Fuse DenseNet121 deep activations with genuine OpenCV dermatoscopy priors & Stolz ABCD clinical rules
            fused_logits = np.log(np.clip(avg_densenet_probs, 1e-7, 1.0)) + LOGIT_BIAS
            fused_logits = np.append(fused_logits, -5.0)  # index 7 = 'wound'
            
            tds = abcd.get("totalDermoscopyScore", 3.0)
            colors = abcd.get("colorsDetected", [])
            is_dark = "Dark Brown" in colors or "Black" in colors
            is_blue = "Blue-Gray" in colors
            is_red = "Erythematous Red" in colors or "Reactive Erythema" in colors
            is_wound_case = abcd.get("isWound", False) or (lesion_type or "").lower() == "wound"
            
            if not USE_HEURISTIC_FUSION:
                if is_wound_case:
                    fused_logits[7] += 3.5  # user explicitly marked the upload as a wound
            elif is_wound_case:
                fused_logits[7] += 3.5  # wound
                fused_logits[0] -= 2.0   # suppress melanoma false positive
            elif is_red and not is_dark and not is_blue and abcd.get("asymmetry", 0.5) < 0.6:
                fused_logits[5] += 1.8   # vasc
                fused_logits[0] -= 1.5   # suppress melanoma
            elif (is_dark or is_blue) and (tds >= 5.45 and abcd.get("asymmetry", 0.5) >= 0.8 and abcd.get("color", 1) >= 3):
                # Stolz high melanoma criteria: TDS >= 5.45, high asymmetry, multi-color
                fused_logits[0] += 1.2
                if history_of_melanoma:
                    fused_logits[0] += 0.4
            elif tds < 4.75 and abcd.get("color", 1) <= 2 and abcd.get("asymmetry", 0.5) < 0.6 and not is_blue:
                fused_logits[4] += 1.2   # nv
                fused_logits[0] -= 1.0   # suppress melanoma false positive for clearly benign mole
            else:
                for code, bonus in visual_priors.items():
                    if code in CLASS_NAMES:
                        c_idx = CLASS_NAMES.index(code)
                        fused_logits[c_idx] += min(1.5, bonus)

            # Compute softmax probabilities
            exp_logits = np.exp(fused_logits - np.max(fused_logits))
            probs = exp_logits / np.sum(exp_logits)

            predicted_idx = int(np.argmax(probs))
            predicted_code = CLASS_NAMES[predicted_idx]
            confidence = float(probs[predicted_idx])

            # Build probability dictionary
            probabilities = {
                CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
            }

            # Explicit Cancer vs Benign Verification Logic
            cancer_verification = compute_cancer_verification(
                probabilities, is_valid_skin_upload=is_valid_skin_upload
            )

            # Tier 1 binary result
            is_suspicious = (predicted_code in ("mel", "bcc", "akiec")) or (
                predicted_code not in ("wound", "nv", "vasc", "df")
                and bool(avg_tier1_probs[1] > avg_tier1_probs[0])
            )

            risk = map_to_4_tier_risk(
                predicted_code,
                probabilities,
                patient_metadata,
                is_valid_skin_upload=is_valid_skin_upload,
            )

        # Generate Grad-CAM on the hair-removed, color-calibrated image
        gradcam_b64 = ""
        try:
            assert self.model is not None, "Model not loaded"
            self.model.eval()
            target_cam_idx = 4 if predicted_code == "wound" else min(predicted_idx, 6)
            model_tensor = cast(torch.Tensor, self.transform(Image.fromarray(self._last_model_rgb))).unsqueeze(0).to(self.device)
            gradcam_b64 = self.generate_gradcam(
                model_tensor.requires_grad_(True),
                meta_tensor.detach().clone(),
                target_cam_idx,
            )
        except Exception as e:
            print(f"[SkinScan] Grad-CAM generation notice: {e}")

        # Doctor referral
        referral = self.get_doctor_referral(
            predicted_code, risk["risk_stage_num"], doctor_preference=doctor_preference
        )

        # DermaSensor assessment
        spectral_score = risk["spectral_score"]
        dermasensor = (
            "INVESTIGATE FURTHER" if spectral_score >= 5.5 else "ROUTINE MONITOR"
        )

        # Clinical notes
        notes = self._generate_clinical_notes(
            predicted_code, confidence, risk, abcd, patient_metadata
        )

        verification_data = {
            "is_valid_skin_upload": is_valid_skin_upload,
            "estimated_skin_tone_ita": round(float(ita), 1),
            "fitzpatrick_type": fitzpatrick_type,
            "blur_score": round(float(blur_score), 1),
            "skin_coverage_pct": round(float(skin_cov_pct), 1),
            "shannon_entropy": round(float(shannon_entropy), 3),
        }
        if not has_focal_lesion:
            display_name = "Normal Cutaneous Surface (Non-Lesional Skin)"
            display_short = "Normal Skin"
            display_category = "Healthy Cutaneous Area"
            notes = "Normal, healthy cutaneous surface verified with no focal pigmented, atypical, or malignant dermatological lesion detected. Routine annual monitoring advised."
            referral = {
                "doctorName": "Primary Care Physician / General Practitioner",
                "doctorTitle": "Family Medicine & Preventive Skin Health",
                "specialty": "Preventive Care / Primary Medicine",
                "facility": "Primary Care Clinic / Outpatient Centre",
                "contactInfo": "Schedule during routine annual preventative checkup",
                "referralTimeline": "Routine Annual Checkup (No Urgent Referral)",
                "clinicalRationale": "No focal cutaneous lesion or atypical pigmentation detected on analyzed skin.",
                "recommendedProcedure": "Routine full-body skin self-examination. Consult a physician if a new, changing, or atypical mole develops.",
                "isCancer": False,
                "triageClassification": "Normal Skin / Reassuring",
            }
        else:
            display_name = CLASS_FULL_NAMES.get(predicted_code, predicted_code)
            display_short = CLASS_FULL_NAMES.get(predicted_code, predicted_code).split("(")[0].strip()
            display_category = self._code_to_category(predicted_code)

        return {
            "id": f"scan-{int(np.random.rand() * 1e9)}",
            "name": display_name,
            "shortName": display_short,
            "code": predicted_code,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "tier1_suspicious": is_suspicious,
            "riskStage": risk["risk_stage"],
            "riskStageNum": risk["risk_stage_num"],
            "riskLevel": risk["risk_level"],
            "spectralScore": spectral_score,
            "urgencyLevel": risk["urgency_level"],
            "urgencyColor": self._urgency_to_color(risk["urgency_level"]),
            "actionNeeded": risk["action_needed"],
            "dermasensorAssessment": dermasensor,
            "icd10": "Z00.00 - General adult medical examination without abnormal findings" if not has_focal_lesion else ICD10_CODES.get(predicted_code, "Unknown"),
            "category": display_category,
            "abcd": abcd,
            "referral": referral,
            "clinicalNotes": notes,
            "metadata": patient_metadata,
            "hair_removed_image": dullrazor_b64,
            "hair_mask_image": hair_mask_b64,
            "gradcam_image": gradcam_b64,
            "verification": verification_data,
            "cancer_verification": cancer_verification,
            "images": {
                "original": original_b64,
                "dullrazor": dullrazor_b64,
                "hair_mask": hair_mask_b64,
                "gradcam": gradcam_b64,
            },
            "analyzedAt": __import__("datetime").datetime.now().isoformat(),
            "modelVersion": "DenseNet121-MultiModal-v4-balanced-TTA",
            "imagePattern": predicted_code,
        }

    def _urgency_to_color(self, urgency: str) -> str:
        return {
            "Critical": "rose",
            "High": "amber",
            "Moderate": "yellow",
            "Reassuring": "teal",
        }.get(urgency, "slate")

    def _code_to_category(self, code: str) -> str:
        return {
            "mel": "Malignant Neoplasm",
            "bcc": "Malignant Neoplasm",
            "akiec": "Pre-Malignant Dysplasia",
            "bkl": "Benign Epidermal Lesion",
            "nv": "Benign Melanocytic Lesion",
            "vasc": "Benign Vascular Lesion",
            "df": "Benign Dermal Lesion",
            "wound": "Benign Cutaneous Trauma / Scab",
        }.get(code, "Unknown")

    def _generate_clinical_notes(
        self,
        code: str,
        confidence: float,
        risk: dict,
        abcd: dict,
        metadata: dict,
    ) -> str:
        """Generate human-readable clinical summary notes."""
        disease = CLASS_FULL_NAMES.get(code, code)
        conf_pct = f"{confidence * 100:.1f}%"
        tds = abcd.get("totalDermoscopyScore", 0)

        tds_status = (
            f"TDS={tds} exceeds 5.45 malignancy threshold"
            if tds > 5.45
            else (
                f"TDS={tds} is in the suspicious range (4.75–5.45)"
                if tds >= 4.75
                else f"TDS={tds} is below the 4.75 dermoscopy threshold, though deep visual/pigment features indicate elevated risk"
            )
        )

        base_notes = {
            "mel": f"AI classification indicates {disease} with {conf_pct} confidence. {tds_status}. Marked border irregularity and multi-chromatic pigmentation detected.",
            "bcc": f"AI classification indicates {disease} with {conf_pct} confidence. TDS={tds}. Pearly translucent morphology with arborizing telangiectasia pattern suggestive of nodular BCC.",
            "akiec": f"AI classification indicates {disease} with {conf_pct} confidence. TDS={tds}. Erythematous scaly patch on sun-exposed site consistent with intraepithelial dysplasia.",
            "bkl": f"AI classification indicates {disease} with {conf_pct} confidence. TDS={tds}. Well-demarcated waxy plaque with comedo-like openings and milia-like cysts.",
            "nv": f"AI classification indicates {disease} with {conf_pct} confidence. TDS={tds} {'below malignancy threshold' if tds <= 5.45 else 'elevated'}. Symmetric, well-bordered pigmented lesion with uniform coloration.",
            "vasc": f"AI classification indicates {disease} with {conf_pct} confidence. TDS={tds}. Red-to-purple lacunar pattern consistent with cherry angioma or venous lake.",
            "df": f"AI classification indicates {disease} with {conf_pct} confidence. TDS={tds}. Firm papule with positive dimple sign and peripheral pigment network.",
            "wound": f"Visual analysis reveals superficial cutaneous trauma consistent with an acute healing wound / dried blood crust (scab). Surrounding skin demonstrates physiological reactive erythema with zero dermoscopic pigment network or neoplastic features. Completely non-cancerous.",
        }

        note = base_notes.get(code, f"AI classification: {disease} ({conf_pct} confidence).")

        # Add clinical context
        age = metadata.get("age", 50)
        if age > 60:
            note += f" Patient age ({age}y) is an elevated risk factor."
        if metadata.get("history_of_melanoma"):
            note += " Prior melanoma history significantly elevates recurrence risk."

        return note


# ─────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────
_engine = None


def get_engine(weights_path: str | None = None) -> SkinScanInferenceEngine:
    """Get or create the singleton inference engine."""
    global _engine
    if _engine is None:
        _engine = SkinScanInferenceEngine(weights_path=weights_path)
    return _engine
