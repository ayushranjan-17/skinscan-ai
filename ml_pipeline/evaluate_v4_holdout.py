"""Evaluate models/skinscan_densenet121_v4.pth on the notebook's lesion-level HAM10000 test split.

Usage (from the project root):
    python3 -I ml_pipeline/evaluate_v4_holdout.py <ham_dir> src/data/modelMetrics.json
<ham_dir> holds HAM10000_metadata.csv and the extracted *.jpg files (searched recursively).
Split, preprocessing and metadata encoding come from holdout_common.py (same as the notebook).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from ml_pipeline.evaluate_metrics import calculate_clinical_metrics, write_model_metrics_json  # noqa: E402
from ml_pipeline.holdout_common import encode_meta, iter_preprocessed, lesion_split, load_ham10000  # noqa: E402
from ml_pipeline.risk_stratification import BIOPSY_ALERT_THRESHOLD  # noqa: E402

ham_dir, out_json = sys.argv[1], sys.argv[2]


# ---- notebook cell 16: model (verbatim) ----
class MultiModalDenseNetHybrid(nn.Module):
    def __init__(self, num_classes=7, meta_dim=16, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.vision_backbone = models.densenet121(weights=weights)
        in_features = self.vision_backbone.classifier.in_features
        self.vision_backbone.classifier = nn.Identity()
        self.metadata_branch = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(inplace=True))
        self.tier1_gating_head = nn.Sequential(
            nn.Linear(in_features + 32, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 2))
        self.tier2_classification_head = nn.Sequential(
            nn.Linear(in_features + 32, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes))

    def forward(self, img, meta, clamp=False):
        m = self.metadata_branch(meta)
        if clamp:  # what backend/inference.py does at serving time
            m = torch.clamp(m, 0.0, 2.5)
        fused = torch.cat((self.vision_backbone(img), m), dim=1)
        return self.tier2_classification_head(fused), self.tier1_gating_head(fused)


df = load_ham10000(ham_dir)
_, test_df = lesion_split(df)
print(f'Test images: {len(test_df)}  class counts: {test_df["dx"].value_counts().to_dict()}', flush=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = MultiModalDenseNetHybrid(pretrained=False)
sd = torch.load(f'{PROJECT}/models/skinscan_densenet121_v4.pth', map_location='cpu', weights_only=True)
model.load_state_dict(sd, strict=True)  # raises if v4 is not this architecture
model.to(device).eval()
print('v4 loaded strictly into the notebook architecture', flush=True)

eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

P, Pc, Y = [], [], []
t0 = time.time()
with torch.no_grad():
    for rows, rgb in iter_preprocessed(test_df, batch=32):
        x = torch.stack([eval_tf(Image.fromarray(a)) for a in rgb]).to(device)
        m = torch.stack([torch.from_numpy(encode_meta(r)) for _, r in rows.iterrows()]).to(device)
        P.append(F.softmax(model(x, m)[0], 1).cpu().numpy())
        Pc.append(F.softmax(model(x, m, clamp=True)[0], 1).cpu().numpy())
        Y.append(rows['label'].to_numpy())
        print(f'  {sum(len(y) for y in Y)}/{len(test_df)}  {time.time() - t0:.0f}s', flush=True)

P, Pc, Y = np.concatenate(P), np.concatenate(Pc), np.concatenate(Y)


def summarise(name, probs):
    ok = np.isfinite(probs).all(1)
    pr, yy = probs[ok], Y[ok]
    m = calculate_clinical_metrics(yy, pr.argmax(1), pr)
    print(f'{name}: n={ok.sum()} (NaN excluded {(~ok).sum()})  acc={m["overall_accuracy"]:.4f}  '
          f'macroAUC={m["macro_roc_auc"]:.4f}  mal_sens={m["malignant_sensitivity_recall"]:.4f}  '
          f'mel_recall={m["classification_report"]["mel"]["recall"]:.3f}', flush=True)
    return m, pr, yy


m_train, pr, yy = summarise('training-time forward (no clamp)', P)
summarise('deployed forward (meta clamp 0..2.5)', Pc)

# The app raises a biopsy alert when mel + bcc + akiec probability reaches the alert threshold.
# Alert figures use the deployed forward pass (metadata features clamped, as in backend/inference.py).
ALERT_THRESHOLD = BIOPSY_ALERT_THRESHOLD
ok_c = np.isfinite(Pc).all(1)
prc, yyc = Pc[ok_c], Y[ok_c]
alerted = (prc[:, 0] + prc[:, 1] + prc[:, 2]) >= ALERT_THRESHOLD
group = np.isin(yyc, [0, 1, 2])
cancers = np.isin(yyc, [0, 1])
extra = {
    'alert_threshold': ALERT_THRESHOLD,
    'cancer_alert_sensitivity': float(alerted[cancers].mean()),   # melanoma + BCC
    'alert_sensitivity': float(alerted[group].mean()),            # melanoma + BCC + actinic keratosis
    'melanoma_alert_sensitivity': float(alerted[yyc == 0].mean()),
    'alert_specificity': float((~alerted[~group]).mean()),        # benign classes not flagged
    'alert_rate': float(alerted.mean()),
    'alert_ppv': float(alerted[group].sum() / max(1, alerted.sum())),
}
try:  # keep the baselines written by evaluate_baselines.py
    with open(out_json) as f:
        extra['comparison'] = json.load(f).get('comparison', [])
except (OSError, ValueError):
    pass
print({k: round(v, 3) for k, v in extra.items() if isinstance(v, float)})

write_model_metrics_json(m_train, yy, pr,
                         'HAM10000, lesion-level split (seed 42, 15% test), notebook preprocessing',
                         'skinscan_densenet121_v4.pth', out_json, extra=extra)
print('wrote', out_json)
