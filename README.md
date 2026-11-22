# SkinScan.AI | Point-of-Care Dermatological Triage Platform

> **Skin lesion risk stratification and explainable AI decision support.** It flags lesions for a closer look. It does not diagnose, and a clinician should make every final call.
> Inspired by [DermaSensor](https://www.dermasensor.com/).
> **Institution:** B.M.S. College of Engineering (BMSCE), Department of CSE
> **Major Project Phase-2:** Akshat Kumar Sinha, Anmol Bhattarai, Ayaan Shrestha, Ayush Ranjan | **Guide:** Megavalli M

---

## How it works

```
+-----------------------------------------------------------------------------------------+
|  INPUT: lesion photo (upload / camera / built-in demo cases) + age, sex, body site      |
+-----------------------------------------------------------------------------------------+
                                             |
+-----------------------------------------------------------------------------------------+
|  PREPROCESSING: the model gets its training preprocessing (DullRazor + gray-world);     |
|  the screen shows a cleaner two-pass hair-removed, skin-tone-calibrated copy;           |
|  Fitzpatrick (ITA) skin-tone and image-quality gate                                     |
+-----------------------------------------------------------------------------------------+
                                             |
+-----------------------------------------------------------------------------------------+
|  MODEL: DenseNet121 image branch (1024-d) + patient-metadata MLP (32-d), fused;         |
|  7 lesion classes, trained with focal loss and cosine annealing with warm restarts      |
+-----------------------------------------------------------------------------------------+
                                             |
+-----------------------------------------------------------------------------------------+
|  DECISION SUPPORT: 1-10 risk score, 4-stage stratification, biopsy alert when the       |
|  combined melanoma + BCC + actinic keratosis probability reaches 17.5%, Grad-CAM        |
|  heatmap, Stolz ABCD scoring, specialist referral, biopsy feedback queue, EHR report    |
+-----------------------------------------------------------------------------------------+
```

---

## Measured results

Evaluated on **1,497 held-out HAM10000 images** from a lesion-level split (15% of lesions, seed 42, so no lesion appears in both training and test), using the same preprocessing as training. Accuracy, AUC and the top-prediction figures come from the model alone; the alert figures come from the app's full analysis path (`ml_pipeline/evaluate_live_path.py`). The numbers live in [`src/data/modelMetrics.json`](src/data/modelMetrics.json) and feed the app's Metrics tab.

| Measure | Result |
| :--- | :--- |
| Accuracy | 82.6% |
| Macro ROC-AUC (one class vs the rest) | 0.949 |
| Melanoma recall, top prediction | 48.8% |
| Melanoma + BCC caught, top prediction | 56.3% |
| **Melanoma + BCC flagged at the 17.5% alert** | **91.8%** (95% CI 87.7% to 94.7%) |
| Melanoma flagged at the alert | 91.7% |
| Benign lesions cleared at the alert | 71.0% |
| Alert fires on | about 41% of all lesions; about 44% of alerts are melanoma, BCC or actinic keratosis |

The top prediction misses many melanomas because the model often ranks them second. The alert is the operating point a clinician actually sees, which is why both are shown. The 17.5% level was first picked on separate validation lesions to reach about 90% melanoma + BCC sensitivity, then re-checked through the app's full analysis path (validation 91.2%, test 91.8%). Picking the level fresh on that path gives 19%, which flags 90.6% and clears 74.0%; 17.5% was kept for its extra margin. It is a trade-off: at 25% the alert flags 84.9% of melanoma + BCC and clears 81.5% of benign lesions, and at 10% it flags 95.9% and clears 57.0%.

The app gives probabilities from the model alone. Earlier versions also applied hand-tuned rule adjustments and replaced the output with fixed numbers when no lesion was found. Neither was validated, and together they cut the share of melanoma + BCC flagged from 91.8% to 63.7% on the same lesions, so both were removed.

Baselines on the same test images (trained on the remaining lesions with balanced class weights):

| Model | Accuracy | Macro AUC |
| :--- | :--- | :--- |
| **SkinScan v4** | **82.6%** | **0.949** |
| ImageNet DenseNet121, not fine-tuned, + logistic regression | 72.7% | 0.910 |
| Patient data only (age, sex, site) + logistic regression | 29.3% | 0.743 |
| Always predicts the most common class (nevus) | 66.8% | 0.500 |

The not-fine-tuned baseline catches more melanoma and BCC on the top prediction (63.3%) because balanced class weights push it to guess rare classes more often, at a large cost in accuracy. Macro AUC is the fairer comparison.

**Limits to keep in mind**
- One split of one public dataset, with no external validation.
- Acral and volar skin (palms, soles, toes) is almost absent from the training data: HAM10000 has 7 acral images, none melanoma. Two DermNet dermoscopy images of an acral melanoma in situ were not flagged (14.5% and 7.2% malignant probability, both under the alert).
- HAM10000 is dermoscopy. Phone photos differ, and results on them have not been measured.
- The model depends on its exact training preprocessing. Feeding it the cleaner hair-removed display image instead cut melanoma + BCC sensitivity from about 90% to 68% at the same threshold, so the app gives the model the training version and shows the cleaner one on screen only.
- Hair removal is imperfect on blurry or fine hair; trim hair around the lesion and photograph in focus.
- Decision support only, not a diagnosis.

To reproduce (needs HAM10000 downloaded and extracted; from the repo root):

```bash
python3 -I ml_pipeline/evaluate_v4_holdout.py <ham10000_dir> src/data/modelMetrics.json
python3 -I ml_pipeline/evaluate_baselines.py  <ham10000_dir> src/data/modelMetrics.json
python3 -I ml_pipeline/evaluate_live_path.py  <ham10000_dir> src/data/modelMetrics.json   # about 25 minutes
```

---

## Repository structure

```
Skin_csncer/
├── src/                          # Web app (React 18, Vite, Tailwind CSS)
│   ├── components/               # Navbar, HeroSection, DiagnosticWorkbench, RiskGaugeCard,
│   │                             # VerificationWorkflowCard, GradCamVisualizer, AbcdAnalysisCard,
│   │                             # ResearchMetricsDashboard, ClinicalReportModal, AuthModal,
│   │                             # ScanHistoryDrawer, HitlFeedbackPanel, GuidedOnboarding, MlPipelineModal
│   ├── data/
│   │   ├── benchmarks.js         # Built-in demo case profiles
│   │   └── modelMetrics.json     # Measured results shown in the Metrics tab
│   ├── utils/
│   │   ├── api.js                # Backend client (auth, scans, analyze, health)
│   │   └── imageProcessing.js    # In-browser fallback processing for demo mode
│   └── App.jsx
├── backend/                      # FastAPI service
│   ├── app.py                    # Endpoints: /api/analyze, /api/health, auth, scans, MLOps queue
│   ├── inference.py              # Model loading, preprocessing, Grad-CAM, referral logic
│   ├── database.py               # SQLite accounts, scan history, biopsy feedback
│   └── requirements.txt
├── ml_pipeline/                  # Machine-learning package
│   ├── dullrazor.py              # Hair removal, skin-tone calibration, ITA gate
│   ├── risk_stratification.py    # 4-stage risk and cancer verification
│   ├── gradcam.py                # Grad-CAM extraction
│   ├── train_hybrid_model.py     # Training script
│   ├── evaluate_metrics.py       # Metric computation and modelMetrics.json writer
│   ├── holdout_common.py         # Split, preprocessing and metadata shared by evaluations
│   ├── evaluate_v4_holdout.py    # Held-out evaluation of the shipped model
│   ├── evaluate_live_path.py     # Alert figures measured through the app's full analysis path
│   └── evaluate_baselines.py     # Baselines on the same test images
├── models/                       # Trained weights (the app loads skinscan_densenet121_v4.pth)
├── notebooks/                    # Training notebooks
├── scripts/                      # verify_demo.py (15 automated checks) and hair-removal tests
└── deploy/huggingface/           # Dockerfile and steps for hosting the backend
```

---

## Run it locally

```bash
# Web app
npm install
npm run dev                       # http://localhost:5173

# Backend (in a second terminal)
pip install -r backend/requirements.txt
python3 backend/app.py            # http://localhost:8000

# Automated checks (model, preprocessing, API, auth, feedback queue)
python3 scripts/verify_demo.py
```

Without the backend the app still opens, but runs in demo mode with built-in cases instead of the real model.

## Deploy

- **Web app:** Vercel. Set the environment variable `VITE_API_URL` to the backend's HTTPS address, then redeploy.
- **Backend:** a Hugging Face Space (Docker). Steps are in [`deploy/huggingface/DEPLOY.md`](deploy/huggingface/DEPLOY.md). On the free tier the Space sleeps when idle and its disk resets on restart, so accounts and scan history are not kept.

---

## Sustainable Development Goals

| Goal | How the project relates |
| :--- | :--- |
| **SDG 3: Good Health** (Target 3.4) | Helps catch skin cancer earlier, at the point of care |
| **SDG 9: Innovation** (Target 9.5) | Combines image and patient data in one model and explains its output with Grad-CAM |
| **SDG 10: Reduced Inequalities** (Target 10.2) | Aims to bring dermatology-style triage to clinics without a specialist nearby |

---

## License and academic attestation

Developed as part of the **B.E. Major Project Phase-2** curriculum at **B.M.S. College of Engineering (BMSCE), Bengaluru** (Autonomous Institution under VTU).
*"Service to Humanity is Service to God"*
