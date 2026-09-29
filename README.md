# SkinScan.AI | Point-of-Care Dermatological Triage Platform

> **Clinical Skin Cancer Risk Stratification & Explainable AI Decision Support System**  
> Inspired by **[DermaSensor™](https://www.dermasensor.com/)**  
> **Institution:** B.M.S. College of Engineering (BMSCE), Department of CSE  
> **Major Project Phase-2:** Akshat Kumar Sinha, Anmol Bhattarai, Ayaan Shrestha, Ayush Ranjan | **Guide:** Megavalli M

---

## 🔬 Project Architecture & Key Highlights

```
+-----------------------------------------------------------------------------------------+
|                                    INPUT LAYER                                          |
|   Dermoscopic Photo (Upload / Camera / 7-Class Benchmarks) + Patient Metadata (Age, Sex, Site) |
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
|                                PREPROCESSING ENGINE                                     |
|         DullRazor (Morphological BlackHat Hair Removal) + Gray-World Color Constancy    |
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
|                           MULTI-MODAL HYBRID DL MODEL                                   |
|   Vision Branch: DenseNet121 (1024-d)  │  Metadata Branch: Patient Context MLP (32-d)   |
|   Loss Function: Focal Loss (γ = 2.0)  │  Optimization: Cosine Annealing + Warm Restarts|
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
|                             CLINICAL DECISION SUPPORT                                   |
|   1. DermaSensor 1.0 - 10.0 Spectral Malignancy Score ("MONITOR" vs "INVESTIGATE FURTHER")|
|   2. 4-Stage Risk Stratification: Stage 1 (Benign) to Stage 4 (Malignant)               |
|   3. Explainable AI: Live Grad-CAM (conv5_block16_2_conv) + Stolz ABCD Rule Scoring     |
|   4. EHR-Ready Printable / PDF Medical Triage Summary                                   |
+-----------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
Skin_csncer/
├── src/                                  # Frontend Application (React 18, Tailwind CSS, Lucide)
│   ├── components/
│   │   ├── Navbar.jsx                    # Header, navigation, audit log trigger
│   │   ├── HeroSection.jsx               # DermaSensor aesthetic hero with live clinical counters
│   │   ├── DiagnosticWorkbench.jsx       # Ingestion: Upload, live camera, 7-class benchmark selector
│   │   ├── RiskGaugeCard.jsx             # 1.0–10.0 Spectral score dial & 4-tier risk matrix
│   │   ├── GradCamVisualizer.jsx         # Live canvas: Raw / DullRazor / Grad-CAM / Contour
│   │   ├── AbcdAnalysisCard.jsx          # Stolz ABCD dermoscopy rule quantification
│   │   ├── ClinicalReportModal.jsx       # Printable hospital EHR Diagnostic Report
│   │   ├── ResearchMetricsDashboard.jsx  # 7-Class Confusion Matrix, ROC-AUC (0.94), SDG 3, 9, 10
│   │   └── MlPipelineModal.jsx           # Model architecture, formulas & code viewer
│   ├── data/
│   │   └── benchmarks.js                 # 7 histopathologically verified HAM10000 cases
│   ├── utils/
│   │   └── imageProcessing.js            # In-browser Canvas processing, DullRazor, Grad-CAM, ABCD
│   ├── App.jsx                           # Application state orchestrator
│   ├── index.css                         # Tailwind CSS, glassmorphism, & print media queries
│   └── main.jsx                          # React DOM entry point
│
├── ml_pipeline/                          # Production Machine Learning Package
│   ├── __init__.py                       # Package exports
│   ├── dullrazor.py                      # Morphological BlackHat hair filter & color constancy
│   ├── risk_stratification.py            # 4-tier risk stratification & DermaSensor score logic
│   ├── train_hybrid_model.py             # Multi-modal DenseNet121 + Metadata MLP with Focal Loss
│   ├── gradcam.py                        # Grad-CAM heatmap extraction module
│   └── evaluate_metrics.py               # Comprehensive clinical evaluation suite
│
├── notebooks/
│   └── SkinScan_Colab_Training.ipynb     # Turnkey Google Colab training notebook
│
├── backend/
│   ├── __init__.py                       # Package exports
│   ├── app.py                            # FastAPI REST microservice
│   └── requirements.txt                  # Python dependencies
│
├── .prettierrc.json                      # Code formatting configuration
├── .gitignore                            # Clean repository ignore list
├── package.json                          # Node project configuration
├── tailwind.config.js                    # DermaSensor clinical theme configuration
└── vite.config.js                        # Vite bundler configuration
```

---

## 🚀 Quick Start Guide

### 1. Run the Web Application
```bash
# Install dependencies
npm install

# Start local development server
npm run dev

# Or build and launch preview
npm run build
npm run preview -- --port 5173
```
*Open [http://localhost:5173](http://localhost:5173) in your browser.*

---

### 2. Format & Lint Code
```bash
npm run format
```

---

### 3. Run Python ML Tests
```bash
python3 ml_pipeline/risk_stratification.py
python3 ml_pipeline/dullrazor.py
python3 ml_pipeline/evaluate_metrics.py
```

---

### 4. Run FastAPI Backend (Optional)
```bash
pip install -r backend/requirements.txt
python3 -m uvicorn backend.app:app --reload --port 8000
```

---

## 🎯 Clinical Performance Targets & SDG Alignments

| Metric | Target / Benchmark | Clinical Significance |
| :--- | :--- | :--- |
| **Malignant Recall (Sensitivity)** | **92.8% - 96.0%** | Minimizes catastrophic false negatives for Melanoma & BCC |
| **Overall Multi-Class Accuracy** | **91.4%** | Balanced performance across all 7 lesion types |
| **Macro ROC-AUC** | **0.942** | High discriminative threshold power |
| **SDG 3 (Good Health)** | Target 3.4 | Early detection raises 5-year melanoma survival to >99% |
| **SDG 9 (Innovation)** | Target 9.5 | Multi-modal neural feature fusion & Explainable AI |
| **SDG 10 (Reduced Inequalities)** | Target 10.2 | Democratizes specialist dermatology access to rural areas |

---

## 📜 License & Academic Attestation
Developed as part of the **B.E. Major Project Phase-2** curriculum at **B.M.S. College of Engineering (BMSCE), Bengaluru** (Autonomous Institution under VTU).  
*"Service to Humanity is Service to God"*
