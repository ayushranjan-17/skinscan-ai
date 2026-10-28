"""
Quantitative hair-removal benchmark (answers "how accurate is the DullRazor step?").

Synthetic hairs with a KNOWN mask (random curved strokes, 1-3 px wide) are drawn on real dermoscopy
images at four contrast levels: the hair darkens the skin by 8 %, 15 %, 30 % or 70 % (8 % = faint vellus
hair, 70 % = dark terminal hair). For each method we report:
  recall per contrast level - fraction of true hair pixels detected
  prec / dice               - mask precision (2-px tolerance) and Dice overlap
  PSNR gain                 - improvement of the restored image over the hairy input (vs. the hair-free original)
  lesion                    - % of lesion pixels altered on the ORIGINAL hair-free images (lesion texture destroyed)
Limitations: synthetic hair; scale/ruler marks present in the originals count as false positives.

Usage:  python scripts/evaluate_hair_removal.py <seed> <folder_with_jpgs>
"""
import sys, glob, os, numpy as np, cv2, importlib
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ml_pipeline import dullrazor as dr_v3
def add_hair(img, n, rng, darken):
    h, w = img.shape[:2]; alpha = np.zeros((h, w), np.float32)
    for _ in range(n):
        p0 = rng.uniform([0, 0], [w, h]); ang = rng.uniform(0, np.pi); L = rng.uniform(0.3, 1.0) * max(h, w)
        p2 = p0 + L * np.array([np.cos(ang), np.sin(ang)]); p1 = (p0 + p2) / 2 + rng.normal(0, L * 0.2, 2)
        t = np.linspace(0, 1, 200)[:, None]; pts = ((1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2).astype(np.int32)
        layer = np.zeros((h, w), np.uint8); cv2.polylines(layer, [pts], False, 255, int(rng.integers(1, 4)), cv2.LINE_AA)
        alpha = np.maximum(alpha, layer / 255.0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)
    out = img * (1 - darken * alpha[..., None])                     # hair = skin darkened by `darken`
    return out.clip(0, 255).astype(np.uint8), alpha > 0.3
def classic(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))
    _,m=cv2.threshold(bh,10,255,cv2.THRESH_BINARY); return cv2.inpaint(img,m,2,cv2.INPAINT_TELEA),m
methods={'classic':classic, 'proposed':dr_v3.apply_dullrazor}
imgs=[cv2.imread(f) for f in sorted(glob.glob(os.path.join(sys.argv[2] if len(sys.argv) > 2 else 'demo_images', '*.jpg')))]
les=[dr_v3.lesion_mask(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)) for i in imgs]
rng=np.random.default_rng(int(sys.argv[1]))
levels=[0.08,0.15,0.3,0.7]
cases={d:[] for d in levels}
for d in levels:
    for im in imgs:
        for _ in range(6): cases[d].append((im,)+add_hair(im,int(rng.integers(3,12)),rng,d))
print('recall per hair darkness (8%,15%,30%,70%) | precision | dice | PSNR gain | lesion px altered on hair-free imgs')
for name,fn in methods.items():
    R=[];Pall=[];Dall=[];G=[]
    for d in levels:
        rr=[]
        for clean,hairy,gt in cases[d]:
            out,m=fn(hairy); m=m>0; tol=cv2.dilate(gt.astype(np.uint8),np.ones((5,5),np.uint8))>0
            r=(m&gt).sum()/max(gt.sum(),1); p=(m&tol).sum()/max(m.sum(),1); rr.append(r); Pall.append(p); Dall.append(2*p*r/max(p+r,1e-9))
            G.append(psnr(clean,out,data_range=255)-psnr(clean,hairy,data_range=255))
        R.append(np.mean(rr))
    lp=np.mean([((fn(i)[1]>0)&l).sum()/max(l.sum(),1)*100 for i,l in zip(imgs,les)])
    print(f"{name:8s} recall {' '.join('%.2f'%x for x in R)} | prec {np.mean(Pall):.2f} | dice {np.mean(Dall):.2f} | PSNR +{np.mean(G):.1f} dB | lesion {lp:.2f}%")
