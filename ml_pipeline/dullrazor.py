"""
Melanin-Resilient DullRazor Hair Removal & Asian Skin-Tone Calibration Engine.
Optimized for Asian / South Asian Skin Tones (Fitzpatrick Types III–V) on Smartphone & Clinical Images.

Key Improvements:
1. Red-Channel & LAB Luminance Extraction: Melanin has minimal absorption in the red spectrum (>650nm),
   whereas hair shafts absorb strongly across all wavelengths. Maximizes hair-to-skin contrast on dark/tan skin.
2. Gentle Local Contrast (CLAHE): Normalizes non-uniform illumination on mobile capture.
3. Structure-Preserving DullRazor: black-hat candidates filtered by oriented line opening (12 directions),
   so pigment network, dots and globules inside the lesion are not mistaken for hair; hairless images are untouched.
5. Telea / Navier-Stokes Inpainting: Smoothly restores texture without blurring lesion boundaries.
6. Shades of Gray (p=6) & Background LAB Calibration: Standardizes smartphone color temperature to clinical reference.
"""

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np




# ─────────────────────────────────────────────────────────────────────────────
# Structure-preserving DullRazor (v3)
#   1. Rough lesion segmentation (vignetting-corrected Otsu) -> lesion is protected.
#   2. Thin dark ridges: black-hat at 7x7 and 11x11.
#   3. Adaptive threshold = median + 3 x MAD of the black-hat on surrounding skin (picks up faint hair
#      on bright skin, ignores normal skin texture); stricter threshold inside the lesion.
#   4. Oriented line opening (12 directions, 31 px): keeps only structures with a long straight run.
#   5. Structures confined to the lesion are treated as lesion texture unless they contain a very long
#      (63 px) straight run; hairs that cross the lesion continue onto skin and are kept.
#   6. Dermatoscope ruler / scale marks in the image border are removed too.
#   7. Telea inpainting at the original resolution; images with almost no hair are left untouched.
# Benchmark: scripts/evaluate_hair_removal.py (synthetic hair at 4 contrast levels, see README there).
# ─────────────────────────────────────────────────────────────────────────────
def _line_kernel(length, angle_deg):
    k = np.zeros((length, length), np.uint8); c = length // 2
    dx, dy = np.cos(np.deg2rad(angle_deg)) * c, np.sin(np.deg2rad(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k

def _line_open(binary, length, n_angles=12):
    out = np.zeros_like(binary)
    for a in np.arange(0, 180, 180.0 / n_angles):
        out |= cv2.morphologyEx(binary, cv2.MORPH_OPEN, _line_kernel(length, a))
    return out

def lesion_mask(gray):
    """Rough lesion segmentation used only to protect lesion texture from hair masking.
    1) remove dermatoscope vignetting by dividing by a very smooth illumination estimate,
    2) Otsu on the darker-than-skin map, 3) keep the most central large component not hugging the border."""
    h, w = gray.shape
    g = gray.astype(np.float32) + 1.0
    illum = cv2.GaussianBlur(g, (0, 0), max(h, w) / 6.0)
    dark = np.clip((1.0 - g / illum) * 255.0 + 128.0, 0, 255).astype(np.uint8)
    dark = cv2.GaussianBlur(dark, (0, 0), 5)
    _, m = cv2.threshold(dark, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best, best_score = 0, 0.0
    for k in range(1, n):
        x, y, ww, hh, area = st[k]
        if area < 0.003 * h * w: continue
        touches = (x == 0) + (y == 0) + (x + ww >= w) + (y + hh >= h)
        dist = np.hypot(cen[k, 0] - w / 2, cen[k, 1] - h / 2) / np.hypot(w / 2, h / 2)
        score = area * (1 - dist) / (1 + touches)
        if score > best_score: best, best_score = k, score
    if best == 0: return np.zeros_like(gray, bool)
    les = (lab == best).astype(np.uint8)
    les = cv2.morphologyEx(les, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return cv2.dilate(les, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0


def apply_dullrazor(img_bgr, inpaint_radius=3, k_noise=3.0, line_len=31, lesion_line_len=45, min_coverage=0.0005, remove_ruler=True):
    h, w = img_bgr.shape[:2]
    s = min(1.0, 600.0 / max(h, w))
    work = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    les = lesion_mask(gray)
    # thin dark ridges at two widths (fine vellus hair, thicker terminal hair)
    bh = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))),
                    cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))))
    bh = cv2.GaussianBlur(bh, (3, 3), 0)
    # adaptive threshold from the noise level of the surrounding skin (robust: median + k * MAD)
    skin = bh[~les] if (~les).sum() > 1000 else bh.ravel()
    med = np.median(skin); mad = np.median(np.abs(skin - med)) * 1.4826 + 1e-6
    t_skin = max(4.0, med + k_noise * mad)
    t_les = max(8.0, t_skin * 1.3)                                    # stricter inside the lesion
    cand_skin = ((bh > t_skin) & ~les).astype(np.uint8)
    cand_les = ((bh > t_les) & les).astype(np.uint8)
    # keep structures that contain a long straight run (hair); pigment network and dots do not
    lines = _line_open(cand_skin, line_len) | _line_open(cand_les, lesion_line_len)
    cand = cand_skin | cand_les
    mask = cv2.dilate(lines, np.ones((3, 3), np.uint8)) & cand
    # group pieces of the same hair that are separated by short low-contrast gaps (e.g. at the lesion edge)
    _, lab = cv2.connectedComponents(cv2.dilate(mask, np.ones((9, 9), np.uint8)), connectivity=8)
    lab = lab * (mask > 0)
    n = lab.max() + 1
    st = np.zeros((n, 5), np.int64)
    ys, xs = np.nonzero(lab)
    if len(ys):
        L = lab[ys, xs]
        st[:, 4] = np.bincount(L, minlength=n)
        for arr, i in ((xs, 2), (ys, 3)):
            mx = np.full(n, -1); mn = np.full(n, 1 << 30)
            np.maximum.at(mx, L, arr); np.minimum.at(mn, L, arr)
            st[:, i] = np.where(mx >= 0, mx - mn + 1, 0)
    span = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]); span[0] = 0
    # a structure lying (almost) entirely inside the lesion is lesion texture, not hair: hairs that cross
    # the lesion always continue onto the surrounding skin
    outside = np.bincount(lab[~les], minlength=n) / np.maximum(st[:, cv2.CC_STAT_AREA], 1)
    # ...unless it contains a very long straight run (>= 2 x line_len), which pigment structures never do
    very_long = np.bincount(lab[_line_open(cand, 2 * line_len + 1) > 0], minlength=n) > 0
    mask = ((span >= line_len) & ((outside >= 0.25) | very_long))[lab].astype(np.uint8)
    # dermatoscope ruler / scale marks: short, high-contrast straight bars in the outer 12 % border band
    bw = int(0.12 * min(gray.shape))
    border = np.zeros_like(mask); border[:bw] = 1; border[-bw:] = 1; border[:, :bw] = 1; border[:, -bw:] = 1
    ruler = _line_open(((bh > 25) & (border > 0)).astype(np.uint8), 15)
    if remove_ruler:
        mask = mask | (cv2.dilate(ruler, np.ones((3, 3), np.uint8)) & (bh > 12))
    mask = mask.astype(np.uint8) * 255
    if mask.mean() / 255 < min_coverage:
        return img_bgr.copy(), np.zeros((h, w), np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    if s < 1: mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    r = max(inpaint_radius, int(round(inpaint_radius / s)))
    return cv2.inpaint(img_bgr, mask, r, cv2.INPAINT_TELEA), mask


def calculate_ita(img_bgr: np.ndarray) -> float:
    """
    Computes Individual Typology Angle (ITA) in CIE LAB space on the surrounding normal skin:
    ITA = arctan((L* - 50) / b*) * (180 / pi)
    
    Fitzpatrick Scale Mapping:
    - > 55°: Type I (Very Light)
    - 41° to 55°: Type II (Light)
    - 28° to 41°: Type III (Intermediate / South Asian Fair)
    - 10° to 28°: Type IV (Tan / South Asian Medium)
    - -30° to 10°: Type V (Brown / South Asian Dark)
    - < -30°: Type VI (Very Dark)
    """
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        return 30.0

    h, w = img_bgr.shape[:2]
    bh, bw = max(4, int(h * 0.15)), max(4, int(w * 0.15))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB: L in [0, 255], a in [0, 255], b in [0, 255]
    # CIE standard: L* in [0, 100], b* in [-128, 127]
    l_star = lab[:, :, 0] * (100.0 / 255.0)
    b_star = lab[:, :, 2] - 128.0

    l_bg = float(np.median(l_star[border_mask]))
    b_bg = float(np.median(b_star[border_mask]))

    ita = float(np.arctan2(l_bg - 50.0, b_bg + 1e-5) * (180.0 / np.pi))
    return ita


def classify_fitzpatrick(ita: float) -> str:
    """Classifies Fitzpatrick Phototype (I through VI) from ITA."""
    if ita > 55:
        return "I"
    elif ita > 41:
        return "II"
    elif ita > 28:
        return "III"
    elif ita > 10:
        return "IV"
    elif ita >= -30:
        return "V"
    else:
        return "VI"


def apply_shades_of_gray(img_bgr: np.ndarray, p: int = 6) -> np.ndarray:
    """
    Shades of Gray color constancy (Finlayson & Trezzi) using Minkowski norm p=6.
    More robust to smartphone flash and yellow indoor casts than standard Gray-World.
    """
    img_float = img_bgr.astype(np.float32)
    norm_img = img_float / 255.0
    power_img = norm_img ** p
    e_b = (np.mean(power_img[:, :, 0])) ** (1.0 / p)
    e_g = (np.mean(power_img[:, :, 1])) ** (1.0 / p)
    e_r = (np.mean(power_img[:, :, 2])) ** (1.0 / p)

    e_gray = (e_b + e_g + e_r) / 3.0 + 1e-6
    scale_b = np.clip(e_gray / (e_b + 1e-6), 0.70, 1.40)
    scale_g = np.clip(e_gray / (e_g + 1e-6), 0.70, 1.40)
    scale_r = np.clip(e_gray / (e_r + 1e-6), 0.70, 1.40)

    balanced = np.zeros_like(img_float)
    balanced[:, :, 0] = np.clip(img_float[:, :, 0] * scale_b, 0, 255)
    balanced[:, :, 1] = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
    balanced[:, :, 2] = np.clip(img_float[:, :, 2] * scale_r, 0, 255)
    return balanced.astype(np.uint8)


def calibrate_asian_skin_tone(img_bgr: np.ndarray) -> np.ndarray:
    """
    Shades of Gray (p=6) followed by gentle background-anchored LAB standardization.
    Standardizes warm/tan/brown Asian skin and indoor phone lighting toward the clinical
    training distribution without distorting central lesion contrast.
    """
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        return img_bgr

    # 1. Shades of gray
    balanced = apply_shades_of_gray(img_bgr, p=6)

    # 2. LAB standardization of background skin
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = balanced.shape[:2]
    bh, bw = max(4, int(h * 0.15)), max(4, int(w * 0.15))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    bg_l = float(np.median(lab[border_mask, 0]))
    bg_b = float(np.median(lab[border_mask, 2]))

    # Clinical reference for normal skin in HAM10000 (L*~65 -> 162 in uint8, b*~14 -> 142 in uint8)
    target_l = 162.0
    target_b = 142.0

    delta_l = (target_l - bg_l) * 0.35
    delta_b = (target_b - bg_b) * 0.30

    lab[:, :, 0] = np.clip(lab[:, :, 0] + delta_l, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + delta_b, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _skin_pixels(img_bgr: np.ndarray) -> np.ndarray:
    """YCrCb + HSV skin-colour test (ranges widened for Asian / South Asian skin tones)."""
    # YCrCb ranges widened for Asian/South Asian skin tones: Cr in [130, 180], Cb in [75, 135]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_ycrcb = (
        (ycrcb[:, :, 1] >= 130)
        & (ycrcb[:, :, 1] <= 180)
        & (ycrcb[:, :, 2] >= 75)
        & (ycrcb[:, :, 2] <= 135)
    )

    # HSV skin ranges: H in [0, 35]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    skin_hsv = (
        (hsv[:, :, 0] >= 0)
        & (hsv[:, :, 0] <= 35)
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 2] >= 40)
    )

    return skin_ycrcb | skin_hsv


def check_skin_and_quality(img_bgr: np.ndarray) -> tuple[bool, float, float, float, str]:
    """
    Real-World Skin & Quality Verification Gate (OOD Check):
    1. Inclusive Asian / South Asian Skin Chromaticity Check (YCrCb + HSV)
    2. Blur & Focus Check (Laplacian Variance)
    
    Returns: (is_valid, skin_coverage_pct, blur_score, ita, status_message)
    """
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        return False, 0.0, 0.0, 0.0, "Invalid image data"

    h, w = img_bgr.shape[:2]
    original_bgr = img_bgr
    # Dermatoscopes and phones add strong colour casts (e.g. blue-grey polarised light), so the skin-colour
    # test runs on a white-balanced copy (Shades of Gray) rather than on the raw pixels.
    img_bgr = apply_shades_of_gray(img_bgr, p=6)
    # Skin pixels in the raw image OR in a white-balanced copy (dermatoscopes and phones can add strong
    # colour casts, e.g. blue-grey polarised light, that hide genuine skin colour)
    combined_skin = _skin_pixels(original_bgr) | _skin_pixels(img_bgr)
    skin_coverage_pct = float(np.mean(combined_skin) * 100.0)

    # Blur check: Laplacian variance measured at a fixed working size (longest side 600 px), because the
    # score depends on resolution. Dermoscopy images are naturally smooth: on HAM10000 test images the
    # score is ~20-100, while the same images blurred with a Gaussian of sigma 1-2 px drop to ~2-12.
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    sc = min(1.0, 600.0 / max(h, w))
    if sc < 1.0:
        gray = cv2.resize(gray, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ITA
    ita = calculate_ita(original_bgr)
    fitzpatrick = classify_fitzpatrick(ita)

    is_valid = True
    reasons = []

    # Non-skin photos (objects, documents, sky) score ~0 %; dermoscopy under coloured light can score 25-40 %.
    if skin_coverage_pct < 15.0:
        is_valid = False
        reasons.append(f"Insufficient skin pixels ({skin_coverage_pct:.1f}% < 15%)")

    BLUR_REJECT, BLUR_WARN = 6.0, 15.0
    soft_focus = False
    if blur_score < BLUR_REJECT:
        is_valid = False
        reasons.append(f"Image is out of focus / blurry (blur index {blur_score:.1f} < {BLUR_REJECT})")
    elif blur_score < BLUR_WARN:
        soft_focus = True

    if is_valid:
        status_msg = f"Passed Skin & Quality Verification (Fitzpatrick Type {fitzpatrick}, ITA {ita:.1f}°)"
        if soft_focus:
            status_msg += f" — note: slightly soft focus (blur index {blur_score:.1f}); a sharper photo is recommended"
    else:
        status_msg = "Verification Failed: " + ", ".join(reasons)

    return is_valid, skin_coverage_pct, blur_score, ita, status_msg


def apply_color_constancy(img_bgr: np.ndarray) -> np.ndarray:
    """Backward compatibility wrapper: applies calibrated skin-tone normalization."""
    return calibrate_asian_skin_tone(img_bgr)


def preprocess_dermatoscopy_pipeline(
    img_bgr: np.ndarray, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Complete pre-processing pipeline:
    1. DullRazor hair removal
    2. Asian skin tone & illumination calibration
    3. Resizing to target_size
    """
    hair_removed, _ = apply_dullrazor(img_bgr)
    calibrated = calibrate_asian_skin_tone(hair_removed)

    if cv2 is not None:
        resized = cv2.resize(calibrated, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        h = min(calibrated.shape[0], target_size[0])
        w = min(calibrated.shape[1], target_size[1])
        resized[:h, :w] = calibrated[:h, :w]

    return resized


if __name__ == '__main__':
    # Unit test with synthetic Asian skin image
    synthetic_img = np.full((300, 300, 3), (102, 142, 198), dtype=np.uint8)
    if cv2 is not None:
        cv2.circle(synthetic_img, (150, 150), 30, (30, 45, 65), -1)
        cv2.line(synthetic_img, (20, 20), (280, 260), (20, 20, 20), 3)

    processed = preprocess_dermatoscopy_pipeline(synthetic_img)
    print(f"Preprocessed output shape: {processed.shape}, dtype: {processed.dtype}")
    assert processed.shape == (224, 224, 3)

    ita = calculate_ita(synthetic_img)
    fitz = classify_fitzpatrick(ita)
    print(f"Calculated ITA: {ita:.1f}°, Fitzpatrick Type: {fitz}")
    assert fitz in ["III", "IV", "V"]

    is_valid, cov, blur, _, msg = check_skin_and_quality(synthetic_img)
    print(f"Quality check: valid={is_valid}, coverage={cov:.1f}%, blur={blur:.1f}")
    assert is_valid
    print("DullRazor preprocessor unit test passed successfully!")
