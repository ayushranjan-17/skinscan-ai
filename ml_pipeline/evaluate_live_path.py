"""Measure the app's real analysis path (backend predict(), every rule included) on the held-out test lesions
and write the biopsy-alert figures into src/data/modelMetrics.json.

Usage (from the project root, after evaluate_v4_holdout.py):
    python3 -I ml_pipeline/evaluate_live_path.py <ham_dir> src/data/modelMetrics.json

Every test melanoma, BCC and actinic keratosis lesion is analysed, plus a fixed random sample of 400 of the
benign-class lesions (nevus, benign keratosis, dermatofibroma, vascular) to keep the run near half an hour. The
alert rate and precision are weighted back up to the full test set. Takes roughly 25 minutes on Apple silicon.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from backend.inference import SkinScanInferenceEngine  # noqa: E402
from ml_pipeline.holdout_common import lesion_split, load_ham10000  # noqa: E402
from ml_pipeline.risk_stratification import BIOPSY_ALERT_THRESHOLD  # noqa: E402

ham_dir, out_json = sys.argv[1], sys.argv[2]
df = load_ham10000(ham_dir)
_, test_df = lesion_split(df)
cancers = test_df[test_df['dx'].isin(['mel', 'bcc'])].reset_index(drop=True)
akiec = test_df[test_df['dx'] == 'akiec'].reset_index(drop=True)
benign_all = test_df[test_df['dx'].isin(['nv', 'bkl', 'df', 'vasc'])]
benign = benign_all.sample(400, random_state=0).reset_index(drop=True)
engine = SkinScanInferenceEngine()


def flagged(frame, label):
    out = []
    for i, r in frame.iterrows():
        age = int(r['age']) if pd.notna(r['age']) else 50
        sex = str(r['sex']).lower() if str(r['sex']).lower() in ('male', 'female') else 'other'
        res = engine.predict(open(r['image_path'], 'rb').read(), age=age, sex=sex,
                             localization=str(r['localization']).lower(), lesion_type='mole')
        out.append(bool(res['cancer_verification']['is_cancer_suspected']))
        if i % 50 == 0:
            print(f'  {label}: {i}/{len(frame)}', flush=True)
    return np.array(out)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


f_c, f_a, f_b = flagged(cancers, 'melanoma+BCC'), flagged(akiec, 'actinic keratosis'), flagged(benign, 'benign sample')
weight = len(benign_all) / len(benign)
flagged_total = f_c.sum() + f_a.sum() + weight * f_b.sum()
alert = {
    'alert_threshold': BIOPSY_ALERT_THRESHOLD,
    'alert_basis': 'live predict() path; all melanoma, BCC and actinic keratosis test lesions, 400 of 1201 benign',
    'cancer_alert_sensitivity': float(f_c.mean()),
    'cancer_alert_sensitivity_ci': wilson(int(f_c.sum()), len(f_c)),
    'melanoma_alert_sensitivity': float(f_c[(cancers['dx'] == 'mel').to_numpy()].mean()),
    'alert_sensitivity': float((f_c.sum() + f_a.sum()) / (len(f_c) + len(f_a))),
    'alert_specificity': float(1 - f_b.mean()),
    'alert_rate': float(flagged_total / len(test_df)),
    'alert_ppv': float((f_c.sum() + f_a.sum()) / flagged_total),
}
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in alert.items()})

with open(out_json) as f:
    metrics = json.load(f)
metrics.update(alert)
with open(out_json, 'w') as f:
    json.dump(metrics, f, indent=2)
print('wrote alert figures into', out_json)
