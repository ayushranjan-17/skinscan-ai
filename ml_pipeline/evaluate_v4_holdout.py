"""Evaluate models/skinscan_densenet121_v4.pth on the notebook's lesion-level HAM10000 test split.

Usage (from the project root):
    python3 -I ml_pipeline/evaluate_v4_holdout.py <ham_dir> src/data/modelMetrics.json
<ham_dir> holds HAM10000_metadata.csv and extracted *.jpg (searched recursively).
Everything below mirrors SkinScan_Retrain_LeakFree.ipynb (cells 5-7, 11, 12-14, 16, 22).
"""
import glob, json, os, sys, time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import models, transforms

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from ml_pipeline.evaluate_metrics import calculate_clinical_metrics, write_model_metrics_json  # noqa: E402

ham_dir, out_json = sys.argv[1], sys.argv[2]
CLASS_NAMES = ['mel', 'bcc', 'akiec', 'bkl', 'nv', 'vasc', 'df']
LABEL_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}
LOCALIZATIONS = ['back', 'face', 'scalp', 'chest', 'abdomen', 'upper extremity', 'lower extremity', 'trunk']
SEXES = ['male', 'female', 'other']


# ---- notebook cell 5: preprocessing (verbatim) ----
def apply_dullrazor(img_bgr, kernel_size=9, inpaint_radius=2):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA), mask


def apply_color_constancy(img_bgr):
    img_float = img_bgr.astype(np.float32)
    b_mean, g_mean, r_mean = (np.mean(img_float[:, :, i]) for i in range(3))
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    for i, m in enumerate((b_mean, g_mean, r_mean)):
        img_float[:, :, i] = np.clip(img_float[:, :, i] * (gray_mean / (m + 1e-6)), 0, 255)
    return img_float.astype(np.uint8)


def preprocess(img_bgr, target_size=(224, 224)):
    hair_removed, _ = apply_dullrazor(img_bgr)
    return cv2.resize(apply_color_constancy(hair_removed), target_size, interpolation=cv2.INTER_AREA)


def encode_meta(row):
    v = np.zeros(16, dtype=np.float32)
    age = row['age']
    age = float(age) if pd.notna(age) and np.isfinite(age) else 50.0
    v[0] = min(age, 100.0) / 100.0
    v[1] = 6.0 / 120.0
    v[2] = 0.0
    s = str(row['sex']).lower()
    v[3 + (SEXES.index(s) if s in SEXES else 0)] = 1.0
    loc = str(row['localization']).lower()
    v[6 + (LOCALIZATIONS.index(loc) if loc in LOCALIZATIONS else 0)] = 1.0
    return v


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


# ---- notebook cells 3, 6: data + lesion-level split ----
df = pd.read_csv(f'{ham_dir}/HAM10000_metadata.csv')
paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f'{ham_dir}/**/*.jpg', recursive=True)}
df['image_path'] = df['image_id'].map(paths)
df = df[df['image_path'].notna()].reset_index(drop=True)
df['label'] = df['dx'].map(LABEL_MAP)
assert len(df) == 10015, f'Expected 10,015 images, found {len(df)}'

lesions = df.groupby('lesion_id')['dx'].first().reset_index()
tr_l, te_l = train_test_split(lesions, test_size=0.15, stratify=lesions['dx'], random_state=42)
test_df = df[df['lesion_id'].isin(te_l['lesion_id'])].reset_index(drop=True)
print(f'Test images: {len(test_df)}  class counts: {test_df["dx"].value_counts().to_dict()}', flush=True)

# ---- model ----
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
BATCH = 32
with torch.no_grad():
    for s in range(0, len(test_df), BATCH):
        chunk = test_df.iloc[s:s + BATCH]
        imgs, metas = [], []
        for _, r in chunk.iterrows():
            bgr = preprocess(cv2.imread(r['image_path']))
            imgs.append(eval_tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
            metas.append(torch.from_numpy(encode_meta(r)))
        x, m = torch.stack(imgs).to(device), torch.stack(metas).to(device)
        P.append(F.softmax(model(x, m)[0], 1).cpu().numpy())
        Pc.append(F.softmax(model(x, m, clamp=True)[0], 1).cpu().numpy())
        Y.append(chunk['label'].to_numpy())
        if (s // BATCH) % 10 == 0:
            print(f'  {s + len(chunk)}/{len(test_df)}  {time.time() - t0:.0f}s', flush=True)

P, Pc, Y = np.concatenate(P), np.concatenate(Pc), np.concatenate(Y)


def summarise(name, probs):
    ok = np.isfinite(probs).all(1)
    pr, yy = probs[ok], Y[ok]
    pred = pr.argmax(1)
    m = calculate_clinical_metrics(yy, pred, pr)
    print(f'{name}: n={ok.sum()} (NaN excluded {(~ok).sum()})  acc={m["overall_accuracy"]:.4f}  '
          f'macroAUC={m["macro_roc_auc"]:.4f}  mal_sens={m["malignant_sensitivity_recall"]:.4f}  '
          f'mel_recall={m["classification_report"]["mel"]["recall"]:.3f}', flush=True)
    return m, pr, yy


m_train, pr, yy = summarise('training-time forward (no clamp)', P)
summarise('deployed forward (meta clamp 0..2.5)', Pc)
print(json.dumps(m_train['classification_report'], indent=1, default=float)[:1800])

# The app raises a biopsy alert when mel + bcc + akiec probability reaches ALERT_THRESHOLD.
ALERT_THRESHOLD = 0.25
malignant_group = np.isin(yy, [0, 1, 2])
alerted = (pr[:, 0] + pr[:, 1] + pr[:, 2]) >= ALERT_THRESHOLD
extra = {
    'alert_threshold': ALERT_THRESHOLD,
    'alert_sensitivity': float(alerted[malignant_group].mean()),
    'alert_specificity': float((~alerted[~malignant_group]).mean()),
    'alert_rate': float(alerted.mean()),
    'melanoma_alert_sensitivity': float(alerted[yy == 0].mean()),
}
print({k: round(v, 3) for k, v in extra.items()})

write_model_metrics_json(m_train, yy, pr,
                         'HAM10000, lesion-level split (seed 42, 15% test), notebook preprocessing',
                         'skinscan_densenet121_v4.pth', out_json, extra=extra)
print('wrote', out_json)
