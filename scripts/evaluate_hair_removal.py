"""
Quantitative evaluation of hair removal (answers "how accurate is DullRazor?").

Hair-free dermoscopy images get synthetic hairs with a KNOWN mask (random Bezier curves, 1-3 px,
dark brown/black, anti-aliased). Each method is scored on:
  recall / precision / dice  - how well the hair mask matches the true hair pixels (2-px tolerance for precision)
  psnr / ssim                - how close the restored image is to the original hair-free image
  psnr_hair                  - PSNR on the hair pixels only (quality of inpainting)
  clean_masked_pct           - % of pixels marked as hair on the ORIGINAL images (lesion texture wrongly removed)
  clean_ssim                 - how much the ORIGINAL images are altered (1.0 = untouched)

Usage:  python scripts/evaluate_hair_removal.py <folder_of_jpgs> <cases_per_image> classic new
"""
import sys, glob, os, numpy as np, cv2, importlib, json
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ml_pipeline.dullrazor import apply_dullrazor as structure_preserving
rng = np.random.default_rng(int(os.environ.get("SEED","0")))

def classic(img, r=2):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    _, m = cv2.threshold(bh, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, m, r, cv2.INPAINT_TELEA), m

def add_hair(img, n, rng):
    h, w = img.shape[:2]
    alpha = np.zeros((h, w), np.float32)
    for _ in range(n):
        p0 = rng.uniform([0, 0], [w, h]); ang = rng.uniform(0, np.pi); L = rng.uniform(0.3, 1.0) * max(h, w)
        p2 = p0 + L * np.array([np.cos(ang), np.sin(ang)]); p1 = (p0 + p2) / 2 + rng.normal(0, L * 0.15, 2)
        t = np.linspace(0, 1, 200)[:, None]
        pts = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)
        layer = np.zeros((h, w), np.uint8)
        cv2.polylines(layer, [pts], False, 255, int(rng.integers(1, 4)), cv2.LINE_AA)
        alpha = np.maximum(alpha, layer / 255.0 * rng.uniform(0.8, 1.0))
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)
    col = np.array([rng.uniform(15, 45), rng.uniform(20, 50), rng.uniform(25, 60)])
    out = img * (1 - alpha[..., None]) + col * alpha[..., None]
    return out.clip(0, 255).astype(np.uint8), (alpha > 0.3)

def evaluate(fn, cases, clean_imgs):
    D, P, Rc, PS, SS, PSh = [], [], [], [], [], []
    for clean, hairy, gt in cases:
        out, m = fn(hairy); m = m > 0
        tol = cv2.dilate(gt.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # 2-px tolerance for precision
        tp_r = (m & gt).sum() / max(gt.sum(), 1); tp_p = (m & tol).sum() / max(m.sum(), 1)
        Rc.append(tp_r); P.append(tp_p); D.append(2 * tp_p * tp_r / max(tp_p + tp_r, 1e-9))
        PS.append(psnr(clean, out, data_range=255)); SS.append(ssim(clean, out, channel_axis=2, data_range=255))
        PSh.append(psnr(clean[gt], out[gt], data_range=255))
    fp, fs = [], []
    for c in clean_imgs:
        out, m = fn(c); fp.append((m > 0).mean() * 100); fs.append(ssim(c, out, channel_axis=2, data_range=255))
    return dict(recall=np.mean(Rc), precision=np.mean(P), dice=np.mean(D), psnr=np.mean(PS), ssim=np.mean(SS),
                psnr_hair=np.mean(PSh), clean_masked_pct=np.mean(fp), clean_ssim=np.mean(fs))

imgs = [cv2.imread(f) for f in sorted(glob.glob(sys.argv[1] + '/*.jpg'))]
cases = []
for im in imgs:
    for k in range(int(sys.argv[2])):
        hairy, gt = add_hair(im, int(rng.integers(3, 15)), rng); cases.append((im, hairy, gt))
base = dict(psnr=np.mean([psnr(c, h, data_range=255) for c, h, _ in cases]), ssim=np.mean([ssim(c, h, channel_axis=2, data_range=255) for c, h, _ in cases]))
print(f"{len(cases)} synthetic-hair cases on {len(imgs)} images | no removal: PSNR {base['psnr']:.2f} SSIM {base['ssim']:.4f}")
res = {}
for name in (sys.argv[3:] or ['classic', 'new']):
    fn = classic if name == 'classic' else structure_preserving
    r = evaluate(fn, cases, imgs); res[name] = r
    print(f"{name:12s} " + " ".join(f"{k} {v:.3f}" for k, v in r.items()))
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in res.items()} | {'baseline': {k: float(v) for k, v in base.items()}}, open('hair_removal_benchmark.json', 'w'), indent=1)
