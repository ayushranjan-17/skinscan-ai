"""
SkinScan AI — Complete Automated QA & Live Demo Verification Suite
Tests all endpoints, parameters, models, and ML pipeline functions end-to-end.
"""

import io
import json
import os
import re
import sys
import time
import unittest
import cv2
import numpy as np
from PIL import Image

# Ensure paths
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "backend"))

import torch
from fastapi.testclient import TestClient
from backend.app import app, engine, startup_load_model
from backend.inference import SkinScanInferenceEngine, MultiModalDenseNetHybrid, LOCALIZATIONS
from ml_pipeline.dullrazor import (
    calculate_ita,
    classify_fitzpatrick,
    apply_shades_of_gray,
    apply_dullrazor,
    check_skin_and_quality,
    lesion_mask,
)
from ml_pipeline.risk_stratification import (
    BIOPSY_ALERT_THRESHOLD,
    MODERATE_RISK_THRESHOLD,
    compute_cancer_verification,
    map_to_4_tier_risk,
)
from ml_pipeline.gradcam import GradCAM


def create_synthetic_image(format="JPEG", mode="RGB", size=(256, 256), draw_lesion=True):
    """Generate in-memory synthetic dermoscopy test image."""
    img = Image.new(mode, size, color=(198, 142, 102) if mode == "RGB" else (198, 142, 102, 255))
    if draw_lesion:
        # Draw a synthetic darker elliptical mole
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([80, 80, 180, 170], fill=(45, 30, 25) if mode == "RGB" else (45, 30, 25, 255))
        # Draw synthetic hair line
        draw.line([30, 40, 220, 210], fill=(15, 15, 15) if mode == "RGB" else (15, 15, 15, 255), width=2)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class TestSkinScanEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("  SKINSCAN.AI — AUTOMATED QA & VERIFICATION SUITE")
        print("=" * 70)
        # Initialize engine and load v4 weights
        weights_v4 = os.path.join(WORKSPACE_ROOT, "models", "skinscan_densenet121_v4.pth")
        cls.weights_path = weights_v4
        cls.client = TestClient(app)

    # ─────────────────────────────────────────────────────────
    # PHASE 1: ENVIRONMENT & MODEL CHECKPOINT
    # ─────────────────────────────────────────────────────────
    def test_01_model_checkpoint_loading(self):
        """Verify models/skinscan_densenet121_v4.pth exists and loads strictly into MultiModalDenseNetHybrid."""
        self.assertTrue(os.path.isfile(self.weights_path), f"Checkpoint missing: {self.weights_path}")
        # Test on CPU
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        use_bn = any(
            k.startswith(("metadata_branch", "tier2_classification_head")) and "running_mean" in k
            for k in state_dict
        )
        model_cpu = MultiModalDenseNetHybrid(
            num_classes=7, meta_dim=16, pretrained=False, use_batch_norm=use_bn
        )
        res = model_cpu.load_state_dict(state_dict, strict=True)
        self.assertEqual(len(res.missing_keys), 0, f"Missing keys in checkpoint: {res.missing_keys}")
        self.assertEqual(len(res.unexpected_keys), 0, f"Unexpected keys in checkpoint: {res.unexpected_keys}")

        # Test on MPS / GPU if available
        if torch.backends.mps.is_available():
            model_mps = model_cpu.to("mps")
            model_mps.eval()
            dummy_img = torch.randn(1, 3, 224, 224, device="mps")
            dummy_meta = torch.randn(1, 16, device="mps")
            with torch.no_grad():
                out2, out1 = model_mps(dummy_img, dummy_meta)
            self.assertEqual(out2.shape, (1, 7))
            self.assertEqual(out1.shape, (1, 2))
            print("  [Pass] Model checkpoint v4 loads strictly on CPU and MPS (Apple Silicon GPU)")
        else:
            print("  [Pass] Model checkpoint v4 loads strictly on CPU")

    # ─────────────────────────────────────────────────────────
    # PHASE 2: PARAMETER & ML PIPELINE AUDIT
    # ─────────────────────────────────────────────────────────
    def test_02_dullrazor_pipeline_and_edge_cases(self):
        """Audit ml_pipeline/dullrazor.py functions and edge cases (grayscale, RGBA, all-black, all-white)."""
        edge_cases = {
            "all_black": np.zeros((120, 120, 3), dtype=np.uint8),
            "all_white": np.full((120, 120, 3), 255, dtype=np.uint8),
            "grayscale_2d": np.full((120, 120), 128, dtype=np.uint8),
            "rgba_4d": np.full((120, 120, 4), (198, 142, 102, 255), dtype=np.uint8),
            "synthetic_skin": np.full((300, 300, 3), (198, 142, 102), dtype=np.uint8),
        }
        for name, img in edge_cases.items():
            ita = calculate_ita(img)
            self.assertIsInstance(ita, float)
            self.assertFalse(np.isnan(ita))

            fitz = classify_fitzpatrick(ita)
            self.assertIn(fitz, ["I", "II", "III", "IV", "V", "VI"])

            sog = apply_shades_of_gray(img, p=6)
            self.assertEqual(sog.ndim, 3)
            self.assertEqual(sog.shape[2], 3)

            dr_img, dr_mask = apply_dullrazor(img)
            self.assertEqual(dr_img.ndim, 3)
            self.assertEqual(dr_mask.ndim, 2)

            lmask = lesion_mask(img)
            self.assertIn(lmask.dtype, [np.uint8])

            valid, cov, blur, ita_ret, msg = check_skin_and_quality(img)
            self.assertIsInstance(valid, bool)
            self.assertIsInstance(msg, str)

        # Verify dermoscopy image with large lesion passes verification (>15% skin, >6.0 blur)
        test_derm = np.full((600, 600, 3), (120, 150, 195), dtype=np.uint8)
        cv2.circle(test_derm, (300, 300), 270, (10, 10, 10), -1)
        valid_derm, cov_derm, blur_derm, _, msg_derm = check_skin_and_quality(test_derm)
        self.assertTrue(valid_derm, f"Dermoscopy with large lesion failed verification: {msg_derm}")
        self.assertGreaterEqual(cov_derm, 15.0)
        self.assertGreaterEqual(blur_derm, 6.0)

        print("  [Pass] DullRazor melanin-resilient functions and edge-case handling validated")

    def test_03_inference_metadata_layout(self):
        """Verify the exact 16-value metadata vector layout in backend/inference.py."""
        engine = SkinScanInferenceEngine(weights_path=self.weights_path, device="cpu")
        meta = engine.encode_metadata(
            age=65,
            sex="female",
            localization="face",
            duration_months=12,
            history_of_melanoma=True,
        ).squeeze(0).cpu().numpy()

        self.assertEqual(len(meta), 16)
        # Index 0: age / 100 clipped to [0, 1]
        self.assertAlmostEqual(meta[0], 0.65, places=4)
        # Index 1: duration constant (6 / 120)
        self.assertAlmostEqual(meta[1], 6.0 / 120.0, places=4)
        # Index 2: melanoma history constant (0.0)
        self.assertEqual(meta[2], 0.0)
        # Indices 3-5: one-hot sex [male, female, other] -> female is index 4
        self.assertEqual(meta[3], 0.0)
        self.assertEqual(meta[4], 1.0)
        self.assertEqual(meta[5], 0.0)
        # Indices 6-13: one-hot site ['back', 'face', 'scalp', 'chest', 'abdomen', 'upper extremity', 'lower extremity', 'trunk']
        # face is index 1 in LOCALIZATIONS -> 6 + 1 = 7
        self.assertEqual(meta[6], 0.0)  # back
        self.assertEqual(meta[7], 1.0)  # face
        # Indices 14-15: reserved zeros
        self.assertEqual(meta[14], 0.0)
        self.assertEqual(meta[15], 0.0)
        print("  [Pass] Exact 16-value metadata encoding layout confirmed")

    def test_04_abcd_features_and_tds_formula(self):
        """Verify Stolz ABCD ranges and TDS = 1.3*A + 0.1*B + 0.5*C + 0.5*min(5, D/2)."""
        engine = SkinScanInferenceEngine(weights_path=self.weights_path, device="cpu")
        synthetic_bgr = np.full((300, 300, 3), (102, 142, 198), dtype=np.uint8)
        import cv2
        cv2.circle(synthetic_bgr, (150, 150), 45, (30, 45, 65), -1)

        abcd, priors = engine.extract_real_dermoscopy_features(synthetic_bgr)
        a = abcd["asymmetry"]
        b = abcd["border"]
        c = abcd["color"]
        d = abcd["diameter"]
        tds = abcd["totalDermoscopyScore"]

        # Check ranges
        self.assertTrue(0.1 <= a <= 2.0, f"A out of range: {a}")
        self.assertTrue(0.5 <= b <= 8.0, f"B out of range: {b}")
        self.assertTrue(1 <= c <= 6, f"C out of range: {c}")
        self.assertTrue(1.0 <= d <= 15.0, f"D out of range: {d}")

        expected_tds = round(float(a * 1.3 + b * 0.1 + c * 0.5 + min(5.0, d / 2.0) * 0.5), 1)
        self.assertAlmostEqual(tds, expected_tds, places=1, msg=f"TDS mismatch: {tds} vs {expected_tds}")
        print("  [Pass] Stolz ABCD ranges and TDS formula verified")

    def test_05_gradcam_hook_lifecycle(self):
        """Verify Grad-CAM attaches cleanly to conv5_block16_2_conv and clears hooks without leaking memory."""
        engine = SkinScanInferenceEngine(weights_path=self.weights_path, device="cpu")
        img_t = torch.randn(1, 3, 224, 224)
        meta_t = torch.randn(1, 16)

        target_layer = engine.model.vision_backbone.features.denseblock4.denselayer16.conv2
        initial_fwd_hooks = len(target_layer._forward_hooks)
        initial_bwd_hooks = len(target_layer._backward_hooks)

        # Run GradCAM 3 times
        for _ in range(3):
            b64 = engine.generate_gradcam(img_t, meta_t, target_class=0)
            self.assertTrue(len(b64) > 100, "GradCAM returned empty base64 string")

        # Confirm hook count returned to baseline
        final_fwd_hooks = len(target_layer._forward_hooks)
        final_bwd_hooks = len(target_layer._backward_hooks)
        self.assertEqual(initial_fwd_hooks, final_fwd_hooks, "Forward hooks leaked!")
        self.assertEqual(initial_bwd_hooks, final_bwd_hooks, "Backward hooks leaked!")
        print("  [Pass] Grad-CAM hook lifecycle and memory cleanup confirmed")

    def test_06_doctor_referral_routing(self):
        """Verify routing for all 4 stages and doctor preferences."""
        engine = SkinScanInferenceEngine(weights_path=self.weights_path, device="cpu")
        # Stage 4 Melanoma -> Surgical Oncologist within 48-72h
        r4 = engine.get_doctor_referral("mel", 4)
        self.assertIn("Cutaneous Surgical Oncologist", r4["doctorName"])
        self.assertIn("48 to 72 Hours", r4["referralTimeline"])

        # Stage 3 -> Dermatologist within 2-4 weeks
        r3 = engine.get_doctor_referral("akiec", 3)
        self.assertIn("Dermatologist", r3["doctorName"])
        self.assertIn("2 to 4 Weeks", r3["referralTimeline"])

        # Stage 1 -> Family Doctor / Primary Care
        r1 = engine.get_doctor_referral("nv", 1)
        self.assertIn("Primary Care Physician", r1["doctorName"])

        # 2-argument signature get_doctor_referral(code, preference)
        r_pref = engine.get_doctor_referral("mel", "pcp")
        self.assertIn("Primary Care Physician", r_pref["doctorName"])
        print("  [Pass] Clinical doctor referral routing logic confirmed")

    def test_07_risk_stratification_and_cancer_verification(self):
        """Verify risk weights, adjustments, clamping, and the biopsy alert."""
        # Cancer verification test
        probs = {"mel": 0.20, "bcc": 0.08, "akiec": 0.02, "nv": 0.70}
        verif = compute_cancer_verification(probs)
        self.assertTrue(verif["is_cancer_suspected"])
        self.assertAlmostEqual(verif["malignant_probability"], 0.30, places=2)

        # High risk mapping test (spectral_score >= 5.5 -> INVESTIGATE FURTHER)
        high_mel_probs = {"mel": 0.45, "bcc": 0.05, "akiec": 0.02, "nv": 0.48}
        meta = {"age": 68, "history_of_melanoma": True, "localization": "back"}
        risk_high = map_to_4_tier_risk("mel", high_mel_probs, meta)
        self.assertEqual(risk_high["risk_stage_num"], 4)
        self.assertEqual(risk_high["dermasensor_assessment"], "INVESTIGATE FURTHER")
        self.assertTrue(1.0 <= risk_high["spectral_score"] <= 10.0)

        # Low risk mapping test (spectral_score < 5.5 -> MONITOR)
        low_nv_probs = {"mel": 0.01, "bcc": 0.01, "akiec": 0.01, "nv": 0.97}
        meta_young = {"age": 25, "history_of_melanoma": False, "localization": "abdomen"}
        risk_low = map_to_4_tier_risk("nv", low_nv_probs, meta_young)
        self.assertEqual(risk_low["risk_stage_num"], 1)
        self.assertEqual(risk_low["dermasensor_assessment"], "MONITOR")

        # Malignant-probability escalation unit test (mel=37.4%, nv=61.1% -> INVESTIGATE FURTHER, stage >= 3, score >= 6.0)
        escalation_probs = {
            "mel": 0.374,
            "bcc": 0.004,
            "akiec": 0.0,
            "nv": 0.611,
            "bkl": 0.004,
            "df": 0.0,
            "vasc": 0.007,
        }
        risk_esc = map_to_4_tier_risk("nv", escalation_probs)
        self.assertEqual(risk_esc["dermasensor_assessment"], "INVESTIGATE FURTHER")
        self.assertGreaterEqual(risk_esc["risk_stage_num"], 3)
        self.assertGreaterEqual(risk_esc["spectral_score"], 6.0)
        self.assertIn("Biopsy", risk_esc["action_needed"])
        print("  [Pass] Risk stratification and cancer verification logic confirmed")

    # ─────────────────────────────────────────────────────────
    # PHASE 3: END-TO-END AUTOMATED API TESTS
    # ─────────────────────────────────────────────────────────
    def test_08_api_health_endpoint(self):
        """Test GET /api/health."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("device", data)
        self.assertIn("last_inference_ms", data)
        print(f"  [Pass] GET /api/health: model_loaded={data['model_loaded']}, device={data['device']}")

    def test_09_api_analyze_multiformat(self):
        """Test POST /api/analyze with RGB JPG, RGBA PNG, and WebP images and verify all required keys."""
        required_keys = [
            "probabilities",
            "code",
            "confidence",
            "risk_stage",
            "risk_tier",
            "spectral_score",
            "malignant_probability",
            "verification",
            "abcd",
            "images",
            "referral",
            "action_needed",
            "inference_time_ms",
        ]

        formats = [
            ("scan.jpg", create_synthetic_image("JPEG", "RGB"), "image/jpeg"),
            ("scan.png", create_synthetic_image("PNG", "RGBA"), "image/png"),
            ("scan.webp", create_synthetic_image("WEBP", "RGB"), "image/webp"),
        ]

        for fname, img_bytes, mime in formats:
            resp = self.client.post(
                "/api/analyze",
                files={"file": (fname, img_bytes, mime)},
                data={
                    "age": "55",
                    "sex": "male",
                    "localization": "back",
                    "duration_months": "6",
                    "history_of_melanoma": "false",
                    "doctor_preference": "auto",
                    "lesion_type": "mole",
                },
            )
            self.assertEqual(resp.status_code, 200, f"Analysis failed for {fname}: {resp.text}")
            data = resp.json()

            for key in required_keys:
                self.assertIn(key, data, f"Missing key '{key}' in response for {fname}")

            # Verify image overlays
            self.assertTrue(len(data["images"]["dullrazor"]) > 50)
            self.assertTrue(len(data["images"]["gradcam"]) > 50)
            self.assertTrue(len(data["images"]["original"]) > 50)
            print(f"  [Pass] POST /api/analyze validated for {fname} (Latency: {data['inference_time_ms']}ms)")

    def test_10_api_error_handling(self):
        """Test invalid inputs return clean 4xx errors instead of 500."""
        # Non-image file
        bad_file = self.client.post(
            "/api/analyze",
            files={"file": ("notes.txt", b"Hello world text data", "text/plain")},
            data={"age": 50},
        )
        self.assertEqual(bad_file.status_code, 400)

        # Empty image
        empty_file = self.client.post(
            "/api/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"age": 50},
        )
        self.assertEqual(empty_file.status_code, 400)

        # Out-of-range age: -5
        bad_age = self.client.post(
            "/api/analyze",
            files={"file": ("scan.jpg", create_synthetic_image(), "image/jpeg")},
            data={"age": -5},
        )
        self.assertEqual(bad_age.status_code, 422)

        # Out-of-range age: 150
        bad_age_high = self.client.post(
            "/api/analyze",
            files={"file": ("scan.jpg", create_synthetic_image(), "image/jpeg")},
            data={"age": 150},
        )
        self.assertEqual(bad_age_high.status_code, 422)

        # Missing file
        missing_file = self.client.post("/api/analyze", data={"age": 50})
        self.assertEqual(missing_file.status_code, 400)
        print("  [Pass] Clean 4xx HTTP error responses verified on invalid inputs")

    def test_11_consecutive_latency_benchmark(self):
        """Measure inference latency over 5 consecutive requests without crash or memory growth."""
        latencies = []
        img_bytes = create_synthetic_image("JPEG", "RGB")

        for i in range(5):
            t0 = time.perf_counter()
            resp = self.client.post(
                "/api/analyze",
                files={"file": (f"scan_{i}.jpg", img_bytes, "image/jpeg")},
                data={"age": 50, "sex": "male", "localization": "back"},
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self.assertEqual(resp.status_code, 200)
            latencies.append(elapsed)

        avg_lat = sum(latencies) / len(latencies)
        print(f"  [Pass] 5 Consecutive Calls: Avg Latency = {avg_lat:.1f}ms (Min: {min(latencies):.1f}ms, Max: {max(latencies):.1f}ms)")
        self.assertTrue(avg_lat < 5000, f"Average latency too high: {avg_lat}ms")

    def test_12_benchmarks_endpoint(self):
        """Test GET /api/benchmarks."""
        resp = self.client.get("/api/benchmarks")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("benchmarks", data)
        self.assertEqual(len(data["benchmarks"]), 7)
        print("  [Pass] GET /api/benchmarks returns all 7 HAM10000 cases")

    # ─────────────────────────────────────────────────────────
    # PHASE 4: AUTHENTICATION, SQLITE PERSISTENCE & HITL QUEUE
    # ─────────────────────────────────────────────────────────
    def test_13_auth_sqlite_flow(self):
        """Verify SQLite user registration, duplicate rejection, login, and bearer auth."""
        unique_email = f"dr.sarah.{int(time.time()*1000)}@skinscan.ai"
        reg_payload = {
            "name": "Dr. Sarah Lin",
            "email": unique_email,
            "password": "clinical_secure_pass123",
            "role": "clinician"
        }
        # Register
        reg_resp = self.client.post("/auth/register", json=reg_payload)
        self.assertEqual(reg_resp.status_code, 200, reg_resp.text)
        reg_data = reg_resp.json()
        self.assertIn("token", reg_data)
        self.assertIn("user", reg_data)
        self.assertEqual(reg_data["user"]["email"], unique_email)
        self.assertEqual(reg_data["user"]["role"], "clinician")
        self.assertTrue(reg_data["user"]["id"].startswith("usr_"))

        # Duplicate register fails
        dup_resp = self.client.post("/auth/register", json=reg_payload)
        self.assertEqual(dup_resp.status_code, 400)

        # Login
        login_resp = self.client.post("/auth/login", json={
            "email": unique_email,
            "password": "clinical_secure_pass123"
        })
        self.assertEqual(login_resp.status_code, 200)
        login_token = login_resp.json()["token"]

        # Wrong password fails
        bad_login = self.client.post("/auth/login", json={
            "email": unique_email,
            "password": "wrong_password"
        })
        self.assertEqual(bad_login.status_code, 401)

        # /auth/me with token
        me_resp = self.client.get("/auth/me", headers={"Authorization": f"Bearer {login_token}"})
        self.assertEqual(me_resp.status_code, 200)
        self.assertEqual(me_resp.json()["user"]["email"], unique_email)

        # /auth/me without token fails
        me_unauth = self.client.get("/auth/me")
        self.assertEqual(me_unauth.status_code, 401)
        print("  [Pass] SQLite Auth flow: Register, duplicate rejection, login, and Bearer token verified")

    def test_14_authenticated_vs_guest_scan(self):
        """Verify authenticated scan persists in SQLite while guest scan stays 100% stateless."""
        # Create user
        user_email = f"clinician.{int(time.time()*1000)}@hospital.org"
        reg_resp = self.client.post("/auth/register", json={
            "name": "Dr. Marcus Vance",
            "email": user_email,
            "password": "vance_password_2026",
            "role": "clinician"
        })
        token = reg_resp.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Authenticated scan persists
        img_bytes = create_synthetic_image("JPEG", "RGB")
        auth_scan = self.client.post(
            "/api/analyze",
            files={"file": ("lesion_auth.jpg", img_bytes, "image/jpeg")},
            data={"age": 62, "sex": "male", "localization": "back"},
            headers=headers
        )
        self.assertEqual(auth_scan.status_code, 200)
        auth_data = auth_scan.json()
        self.assertIsNotNone(auth_data.get("scan_id"))
        self.assertTrue(auth_data.get("is_persisted"))
        scan_id = auth_data["scan_id"]
        self.assertTrue(scan_id.startswith("scan_"))

        # Verify scan appears in /scans
        scans_resp = self.client.get("/scans", headers=headers)
        self.assertEqual(scans_resp.status_code, 200)
        user_scans = scans_resp.json()["scans"]
        self.assertEqual(len(user_scans), 1)
        self.assertEqual(user_scans[0]["id"], scan_id)
        self.assertEqual(user_scans[0]["patient_age"], 62)

        # 2. Guest scan does not persist and stays stateless
        guest_scan = self.client.post(
            "/api/analyze",
            files={"file": ("lesion_guest.jpg", img_bytes, "image/jpeg")},
            data={"age": 45, "sex": "female", "localization": "face"}
        )
        self.assertEqual(guest_scan.status_code, 200)
        guest_data = guest_scan.json()
        self.assertIsNone(guest_data.get("scan_id"))
        self.assertFalse(guest_data.get("is_persisted"))

        # User's scans should still be exactly 1
        scans_resp2 = self.client.get("/scans", headers=headers)
        self.assertEqual(len(scans_resp2.json()["scans"]), 1)
        print("  [Pass] Authenticated scan persists to SQLite; Guest scan remains 100% stateless")

    def test_15_hitl_biopsy_feedback_and_mlops_queue(self):
        """Verify clinician biopsy ground truth patching, queue increment, and anonymized JSON export."""
        # Create user & scan
        user_email = f"pathologist.{int(time.time()*1000)}@lab.org"
        reg_resp = self.client.post("/auth/register", json={
            "name": "Dr. Elena Rostova",
            "email": user_email,
            "password": "pathology_lab_pass",
            "role": "clinician"
        })
        token = reg_resp.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        img_bytes = create_synthetic_image("JPEG", "RGB")
        scan_resp = self.client.post(
            "/api/analyze",
            files={"file": ("biopsy_case.jpg", img_bytes, "image/jpeg")},
            data={"age": 55, "sex": "female", "localization": "lower extremity"},
            headers=headers
        )
        scan_id = scan_resp.json()["scan_id"]

        # Initial queue count
        q_init = self.client.get("/mlops/retrain-queue").json()["queued_count"]

        # Submit valid biopsy feedback
        feedback_resp = self.client.patch(
            f"/scans/{scan_id}/feedback",
            json={
                "confirmed_biopsy_label": "mel",
                "clinician_notes": "Histopathology verified: Melanoma in situ (Clark Level I)"
            },
            headers=headers
        )
        self.assertEqual(feedback_resp.status_code, 200)
        fb_data = feedback_resp.json()
        self.assertEqual(fb_data["status"], "success")
        self.assertEqual(fb_data["scan"]["confirmed_biopsy_label"], "mel")
        self.assertEqual(fb_data["scan"]["queued_for_retrain"], 1)

        # Invalid label rejected
        bad_fb = self.client.patch(
            f"/scans/{scan_id}/feedback",
            json={"confirmed_biopsy_label": "not_a_valid_label"},
            headers=headers
        )
        self.assertEqual(bad_fb.status_code, 422)

        # Queue count incremented
        q_after = self.client.get("/mlops/retrain-queue").json()
        self.assertEqual(q_after["queued_count"], q_init + 1)
        matching_case = next((c for c in q_after["cases"] if c["scan_id"] == scan_id), None)
        self.assertIsNotNone(matching_case)
        self.assertEqual(matching_case["confirmed_biopsy_label"], "mel")

        # Export anonymized retraining manifest
        exp_resp = self.client.get("/mlops/export-queue")
        self.assertEqual(exp_resp.status_code, 200)
        exp_data = exp_resp.json()
        self.assertIn("manifest", exp_data)
        manifest_item = next((m for m in exp_data["manifest"] if m["sample_id"] == scan_id), None)
        self.assertIsNotNone(manifest_item)
        self.assertEqual(manifest_item["ground_truth_label"], "mel")

        # Delete scan
        del_resp = self.client.delete(f"/scans/{scan_id}", headers=headers)
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.json()["status"], "deleted")

        # Second delete gives 404
        del_404 = self.client.delete(f"/scans/{scan_id}", headers=headers)
        self.assertEqual(del_404.status_code, 404)
        print("  [Pass] HITL Biopsy feedback, MLOps retrain queue, anonymized JSON export, and scan deletion verified")


    def test_16_alert_threshold_single_source(self):
        """The Python alert level, the front-end constant and the metrics file must agree, and the alert must switch at it."""
        with open(os.path.join(WORKSPACE_ROOT, "src", "data", "alertThreshold.js")) as f:
            js_value = float(re.search(r"BIOPSY_ALERT_THRESHOLD = ([0-9.]+);", f.read()).group(1))
        self.assertEqual(js_value, BIOPSY_ALERT_THRESHOLD)
        with open(os.path.join(WORKSPACE_ROOT, "src", "data", "modelMetrics.json")) as f:
            self.assertEqual(json.load(f)["alert_threshold"], BIOPSY_ALERT_THRESHOLD)

        def probs_with_malignant(m):
            return {"mel": m * 0.6, "bcc": m * 0.3, "akiec": m * 0.1, "nv": 1.0 - m}

        below = compute_cancer_verification(probs_with_malignant(BIOPSY_ALERT_THRESHOLD - 0.02))
        above = compute_cancer_verification(probs_with_malignant(BIOPSY_ALERT_THRESHOLD + 0.02))
        self.assertFalse(below["is_cancer_suspected"])
        self.assertTrue(above["is_cancer_suspected"])
        calm = compute_cancer_verification(probs_with_malignant(MODERATE_RISK_THRESHOLD - 0.01))
        self.assertEqual(calm["risk_tier"], "Low / Reassuring")
        print(f"  [Pass] Alert threshold {BIOPSY_ALERT_THRESHOLD} is identical in Python, the front end and the metrics file")


    def test_17_no_hardcoded_output_when_no_lesion_found(self):
        """A photo with no distinct lesion must still return the model's real probabilities, never fixed numbers."""
        resp = self.client.post(
            "/api/analyze",
            files={"file": ("plain.jpg", create_synthetic_image("JPEG", "RGB", draw_lesion=False), "image/jpeg")},
            data={"age": "55", "sex": "male", "localization": "back", "lesion_type": "mole"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertFalse(data["abcd"]["hasFocalLesion"], "synthetic plain skin should have no focal lesion")
        probs = data["probabilities"]
        # The removed override used these exact numbers (nv 0.985, mel 0.001) and malignant 0.003.
        self.assertFalse(probs["nv"] == 0.985 and probs["mel"] == 0.001, "output was replaced by fixed numbers")
        self.assertNotEqual(data["malignant_probability"], 0.003)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=1)
        self.assertNotIn("Normal Cutaneous Surface", data["name"])
        self.assertIn("No distinct lesion", data["clinicalNotes"])
        print("  [Pass] No-lesion photo returns real model probabilities, not a hard-coded 'normal skin' result")


    def test_18_actinic_keratosis_is_stage_3_not_4(self):
        """Stage 4 is melanoma / BCC. A lesion that is mostly actinic keratosis is Stage 3 even when the total
        malignant-group probability is high; Stage 4 needs the invasive cancers (mel + bcc) themselves."""
        ak = {"mel": 0.003, "bcc": 0.147, "akiec": 0.686, "nv": 0.078, "bkl": 0.078, "df": 0.001, "vasc": 0.0, "wound": 0.007}
        risk = map_to_4_tier_risk("akiec", ak)
        self.assertEqual(risk["risk_stage_num"], 3)
        self.assertTrue(compute_cancer_verification(ak)["is_cancer_suspected"], "the biopsy alert must still fire")

        mel = {"mel": 0.60, "bcc": 0.02, "akiec": 0.05, "nv": 0.30, "bkl": 0.03}
        self.assertEqual(map_to_4_tier_risk("mel", mel)["risk_stage_num"], 4)
        bcc_plus_ak = {"mel": 0.02, "bcc": 0.45, "akiec": 0.20, "nv": 0.30, "bkl": 0.03}
        self.assertEqual(map_to_4_tier_risk("bcc", bcc_plus_ak)["risk_stage_num"], 4)
        print("  [Pass] Actinic keratosis maps to Stage 3; Stage 4 still needs melanoma or BCC")


if __name__ == "__main__":
    unittest.main(verbosity=2)
