"""Baselines scored on the same held-out HAM10000 test images as the v4 model.

Usage (from the project root, after evaluate_v4_holdout.py has written the JSON):
    python3 -I ml_pipeline/evaluate_baselines.py <ham_dir> src/data/modelMetrics.json

Each baseline is trained on the non-test lesions with default settings, then scored on the test set:
  1. always predicts the most common class
  2. patient data only (age, sex, body site) with logistic regression
  3. ImageNet-pretrained DenseNet121 features, no fine-tuning, with logistic regression
The results go into the "comparison" list of the JSON, which the Model comparison tab displays.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from ml_pipeline.evaluate_metrics import calculate_clinical_metrics  # noqa: E402
from ml_pipeline.holdout_common import encode_meta, iter_preprocessed, lesion_split, load_ham10000  # noqa: E402

warnings.filterwarnings('ignore')
ham_dir, out_json = sys.argv[1], sys.argv[2]

df = load_ham10000(ham_dir)
train_df, test_df = lesion_split(df)
y_train, y_test = train_df['label'].to_numpy(), test_df['label'].to_numpy()
print(f'train {len(train_df)}  test {len(test_df)}', flush=True)

rows = []


def score(name, source, probs):
    m = calculate_clinical_metrics(y_test, probs.argmax(1), probs)
    row = {'name': name, 'accuracy': m['overall_accuracy'], 'malignant_recall': m['malignant_sensitivity_recall'],
           'auc': m['macro_roc_auc'], 'source': source}
    print(f'{name}: acc={row["accuracy"]:.4f}  mal_recall={row["malignant_recall"]:.4f}  auc={row["auc"]}', flush=True)
    rows.append(row)


# 1. most common class
prior = np.bincount(y_train, minlength=7) / len(y_train)
score('Always predicts the most common class (nevus)',
      'No learning from the image or patient data', np.tile(prior, (len(test_df), 1)))

# 2. patient data only
X_train = np.stack([encode_meta(r) for _, r in train_df.iterrows()])
X_test = np.stack([encode_meta(r) for _, r in test_df.iterrows()])
meta_lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X_train, y_train)
score('Patient data only (logistic regression)',
      'Age, sex and body site, trained on the other lesions', meta_lr.predict_proba(X_test))

# 3. frozen ImageNet DenseNet121 features + logistic regression
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
backbone.classifier = nn.Identity()
backbone.to(device).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def features(frame, label):
    out, t0 = [], time.time()
    with torch.no_grad():
        for rows_, rgb in iter_preprocessed(frame, batch=64):
            x = torch.stack([tf(Image.fromarray(a)) for a in rgb]).to(device)
            out.append(backbone(x).cpu().numpy())
            print(f'  {label}: {sum(len(o) for o in out)}/{len(frame)}  {time.time() - t0:.0f}s', flush=True)
    return np.concatenate(out)


F_train, F_test = features(train_df, 'train'), features(test_df, 'test')
scaler = StandardScaler().fit(F_train)
feat_lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(scaler.transform(F_train), y_train)
score('ImageNet DenseNet121, not fine-tuned (logistic regression)',
      'Frozen ImageNet features with the same preprocessing, trained on the other lesions',
      feat_lr.predict_proba(scaler.transform(F_test)))

with open(out_json) as f:
    metrics = json.load(f)
metrics['comparison'] = rows
with open(out_json, 'w') as f:
    json.dump(metrics, f, indent=2)
print('wrote comparison into', out_json)
