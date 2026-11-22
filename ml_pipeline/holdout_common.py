"""Shared pieces of the leak-free notebook's HAM10000 evaluation: split, preprocessing, metadata.

Everything here mirrors SkinScan_Retrain_LeakFree.ipynb so the held-out test set, the image
preprocessing and the metadata encoding are identical across the evaluation scripts.
"""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['mel', 'bcc', 'akiec', 'bkl', 'nv', 'vasc', 'df']
LABEL_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}
LOCALIZATIONS = ['back', 'face', 'scalp', 'chest', 'abdomen', 'upper extremity', 'lower extremity', 'trunk']
SEXES = ['male', 'female', 'other']


def _dullrazor(img_bgr, kernel_size=9, inpaint_radius=2):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)


def _color_constancy(img_bgr):
    img_float = img_bgr.astype(np.float32)
    b_mean, g_mean, r_mean = (np.mean(img_float[:, :, i]) for i in range(3))
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    for i, m in enumerate((b_mean, g_mean, r_mean)):
        img_float[:, :, i] = np.clip(img_float[:, :, i] * (gray_mean / (m + 1e-6)), 0, 255)
    return img_float.astype(np.uint8)


def preprocess(img_bgr, target_size=(224, 224)):
    return cv2.resize(_color_constancy(_dullrazor(img_bgr)), target_size, interpolation=cv2.INTER_AREA)


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


def load_ham10000(ham_dir):
    df = pd.read_csv(f'{ham_dir}/HAM10000_metadata.csv')
    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(f'{ham_dir}/**/*.jpg', recursive=True)}
    df['image_path'] = df['image_id'].map(paths)
    df = df[df['image_path'].notna()].reset_index(drop=True)
    df['label'] = df['dx'].map(LABEL_MAP)
    assert len(df) == 10015, f'Expected 10,015 images, found {len(df)}'
    return df


def lesion_split(df):
    """Notebook split: 15% of lesions (stratified, seed 42) are the test set; the rest is train+val."""
    lesions = df.groupby('lesion_id')['dx'].first().reset_index()
    _, test_lesions = train_test_split(lesions, test_size=0.15, stratify=lesions['dx'], random_state=42)
    is_test = df['lesion_id'].isin(test_lesions['lesion_id'])
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def iter_preprocessed(frame, batch=64):
    """Yield (rows, RGB uint8 array [B, 224, 224, 3]) with the notebook preprocessing applied."""
    cv2.setNumThreads(1)

    def one(path):
        return cv2.cvtColor(preprocess(cv2.imread(path)), cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for s in range(0, len(frame), batch):
            rows = frame.iloc[s:s + batch]
            yield rows, np.stack(list(pool.map(one, rows['image_path'])))
