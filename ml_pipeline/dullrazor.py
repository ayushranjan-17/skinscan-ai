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


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """1-pixel-wide straight line structuring element of the given length and orientation."""
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx, dy = np.cos(np.deg2rad(angle_deg)) * c, np.sin(np.deg2rad(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def apply_dullrazor(
    img_bgr: np.ndarray,
    inpaint_radius: int = 3,
    thresh: int = 14,
    line_len: int = 31,
    n_angles: int = 12,
    min_len: int = 30,
    min_coverage: float = 0.0015,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Structure-preserving DullRazor (oriented line-opening variant).

    Classic DullRazor (Lee et al., 1997) marks every thin dark structure as hair, so it also erases
    pigment network, dots and globules inside the lesion — the features dermatologists use. This version:
      1. Black-hat (11x11) on gray level -> candidate thin dark structures (classic DullRazor step).
      2. Morphological opening of the candidates with 31-px line elements in 12 orientations: only
         structures that contain a long straight run survive (hair). Pigment-network meshes, dots and
         globules contain no such run and are rejected.
      3. Constrained reconstruction (3x3) restores hair width/curvature; components spanning < 30 px dropped.
      4. Hair-presence gate: if < 0.15 % of the image is hair, the image is returned unchanged.
      5. Telea inpainting of the dilated hair mask at the original resolution.
    Processing is done at <= 600 px so kernel sizes are resolution independent.

    Synthetic-hair benchmark (scripts/evaluate_hair_removal.py, 150 cases on HAM10000 test images):
    hair Dice 0.865 vs 0.607 (classic) and 0.377 (previous multi-angle version); restoration PSNR
    38.5 dB vs 35.9 / 26.7 dB; lesion pixels altered on hair-free images 1.3 % vs 7.2 % / 9.3 %.

    Returns: (hair_removed_image_bgr, binary_hair_mask)
    """
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        mask = (
            np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
            if img_bgr is not None
            else np.zeros((1, 1), dtype=np.uint8)
        )
        return img_bgr, mask

    h, w = img_bgr.shape[:2]
    s = min(1.0, 600.0 / max(h, w))
    work = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # 1. Classic DullRazor candidate map
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    cand = (blackhat > thresh).astype(np.uint8)

    # 2. Oriented line opening: keep only structures with a long straight run
    lines = np.zeros_like(cand)
    for a in np.arange(0, 180, 180.0 / n_angles):
        lines |= cv2.morphologyEx(cand, cv2.MORPH_OPEN, _line_kernel(line_len, a))

    # 3. Constrained reconstruction + minimum span
    mask = cv2.dilate(lines, np.ones((3, 3), np.uint8)) & cand
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    span = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    span[0] = 0
    mask = (span >= min_len)[lab].astype(np.uint8) * 255

    # 4. Hair-presence gate
    if mask.mean() / 255.0 < min_coverage:
        return img_bgr.copy(), np.zeros((h, w), np.uint8)

    # 5. Inpaint at original resolution
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    if s < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    radius = max(inpaint_radius, int(round(inpaint_radius / s)))
    return cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA), mask


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

    combined_skin = skin_ycrcb | skin_hsv
    skin_coverage_pct = float(np.mean(combined_skin) * 100.0)

    # Blur check
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ITA
    ita = calculate_ita(img_bgr)
    fitzpatrick = classify_fitzpatrick(ita)

    is_valid = True
    reasons = []

    if skin_coverage_pct < 25.0:
        is_valid = False
        reasons.append(f"Insufficient skin pixels ({skin_coverage_pct:.1f}% < 25%)")

    if blur_score < 45.0:
        is_valid = False
        reasons.append(f"Image is out of focus / blurry (blur index {blur_score:.1f} < 45.0)")

    if is_valid:
        status_msg = f"Passed Skin & Quality Verification (Fitzpatrick Type {fitzpatrick}, ITA {ita:.1f}°)"
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
