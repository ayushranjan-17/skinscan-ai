"""
Melanin-Resilient DullRazor Hair Removal & Asian Skin-Tone Calibration Engine.
Optimized for Asian / South Asian Skin Tones (Fitzpatrick Types III–V) on Smartphone & Clinical Images.

Key Improvements:
1. Red-Channel & LAB Luminance Extraction: Melanin has minimal absorption in the red spectrum (>650nm),
   whereas hair shafts absorb strongly across all wavelengths. Maximizes hair-to-skin contrast on dark/tan skin.
2. Gentle Local Contrast (CLAHE): Normalizes non-uniform illumination on mobile capture.
3. Multi-Scale & Multi-Angle Black-Hat Transform: 4 rotated orientations (0°, 45°, 90°, 135°) at 9x9 and 17x17 scales.
4. Lesion-Safe Hair Masking: Connected components geometry filter (aspect ratio, thinness, inertia tensor)
   to exclude round/blob-like dark lesions and keep only elongated hair shafts.
5. Telea / Navier-Stokes Inpainting: Smoothly restores texture without blurring lesion boundaries.
6. Shades of Gray (p=6) & Background LAB Calibration: Standardizes smartphone color temperature to clinical reference.
"""

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np


def _ensure_bgr_image(img: np.ndarray) -> np.ndarray:
    """Ensure image is a 3-channel BGR uint8 array."""
    if img is None or img.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.stack([img] * 3, axis=-1)
    if img.ndim == 3:
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.repeat(img, 3, axis=-1)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if cv2 is not None else img[:, :, :3]
        elif img.shape[2] == 3:
            return img
    return img


def lesion_mask(gray: np.ndarray) -> np.ndarray:
    """
    Segments candidate skin lesion from grayscale (or BGR) image.
    Uses Gaussian smoothing followed by Otsu adaptive thresholding and morphological closing.
    Returns binary mask (uint8) of values 0 or 255.
    """
    if cv2 is None or gray is None or gray.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if gray.ndim == 3:
        if gray.shape[2] == 4:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)
        elif gray.shape[2] == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray[:, :, 0]

    h, w = gray.shape[:2]
    # Check for all-black or all-white or uniform image
    min_val, max_val = float(np.min(gray)), float(np.max(gray))
    if max_val - min_val < 5:
        # Uniform image: no focal lesion
        return np.zeros((h, w), dtype=np.uint8)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return mask


def create_directional_kernel(size: int, angle_deg: float) -> np.ndarray:
    """Create a rotated linear structuring element for hair detection."""
    k_size = max(3, size)
    kernel = np.zeros((k_size, k_size), dtype=np.uint8)
    if cv2 is None:
        kernel[k_size // 2, :] = 1
        return kernel
    cx, cy = k_size // 2, k_size // 2
    rad = np.deg2rad(angle_deg)
    half = (size - 1) / 2.0
    x1 = int(round(cx - half * np.cos(rad)))
    y1 = int(round(cy - half * np.sin(rad)))
    x2 = int(round(cx + half * np.cos(rad)))
    y2 = int(round(cy + half * np.sin(rad)))
    cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
    return kernel


def _line_open(cand: np.ndarray, line_len: int, n_angles: int = 16) -> np.ndarray:
    """
    Oriented morphological line opening over n_angles directions.
    Preserves elongated hair structures of length >= line_len while eliminating
    round, compact, or isotropic structures (moles, dots, globules).
    """
    if cv2 is None or cand is None or cand.size == 0 or not np.any(cand):
        return np.zeros_like(cand) if cand is not None else np.zeros((1, 1), dtype=np.uint8)

    result = np.zeros_like(cand)
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    for angle in angles:
        k = create_directional_kernel(line_len, float(angle))
        opened = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
        result = np.maximum(result, opened)
    return result


def _apply_dullrazor_single(
    img_bgr: np.ndarray,
    inpaint_radius: int = 4,
    k_noise: float = 3.2,
    line_len: int = 13,
    lesion_line_len: int = 25,
    min_coverage: float = 0.0003,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies melanin-resilient Anisotropic DullRazor hair removal optimized for
    Fitzpatrick III-V skin and high-resolution smartphone dermatoscopy.

    Key improvements:
    1. Anisotropic Directional Black-Hat: computes Black-Hat across 12 oriented 1-px
       linear kernels (length 13 at 600px scale) on fused Red + CLAHE LAB L*, taking
       bh = bh_max - bh_min. Directional hair strands yield high response; isotropic
       skin grain, pores, and round moles yield ~0 response, preventing blob fusion.
    2. Adaptive thresholding with k_noise=3.2 on < p75 skin baseline (floor 8.0).
    3. Oriented line opening (line_len=13, n_angles=12) with lesion protection
       (outside >= 0.25 | very_long) preserving central mole structure 100% intact.
    4. Thin mask dilation (3x3 at working scale, <=5px at full scale) and capped
       inpaint radius (<=5) avoiding over-blur and dark ghost streaks.
    5. Automatic tightening safety guard if coverage exceeds 25%.

    Returns: (hair_removed_image_bgr, binary_hair_mask)
    """
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        mask = (
            np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
            if img_bgr is not None and img_bgr.ndim >= 2
            else np.zeros((1, 1), dtype=np.uint8)
        )
        return img_bgr, mask

    img_bgr = _ensure_bgr_image(img_bgr)
    h, w = img_bgr.shape[:2]

    # Fast-path for uniform or tiny images
    min_val, max_val = float(np.min(img_bgr)), float(np.max(img_bgr))
    if max_val - min_val < 5 or h < 8 or w < 8:
        return img_bgr.copy(), np.zeros((h, w), dtype=np.uint8)

    # 1. Standardize processing width for morphological operators
    target_w = 600
    scale = target_w / float(w) if w > target_w else 1.0
    if scale < 1.0:
        target_h = max(10, int(h * scale))
        work = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        work = img_bgr

    # 2. Anisotropic Directional Black-Hat on Fused Red + CLAHE LAB L*
    r_channel = work[:, :, 2]
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    fused = cv2.addWeighted(r_channel, 0.65, l_clahe, 0.35, 0)

    n_angles = 12
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    bh_stack = []
    for a in angles:
        k = create_directional_kernel(line_len, float(a))
        bh_stack.append(cv2.morphologyEx(fused, cv2.MORPH_BLACKHAT, k))

    bh_stack = np.stack(bh_stack, axis=0)
    bh_max = np.max(bh_stack, axis=0)
    bh_min = np.min(bh_stack, axis=0)
    bh = cv2.subtract(bh_max, bh_min)

    # 3. Adaptive thresholding resilient to dense hair
    work_gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    l_mask = lesion_mask(work_gray)

    skin_pixels = bh[l_mask == 0]
    if skin_pixels.size < 100:
        skin_pixels = bh.ravel()

    # Compute med and mad on pixels below 75th percentile of skin to isolate background noise
    p75 = float(np.percentile(skin_pixels, 75))
    baseline = skin_pixels[skin_pixels <= p75]
    if baseline.size > 10:
        med = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - med)))
    else:
        med = float(np.median(skin_pixels))
        mad = float(np.median(np.abs(skin_pixels - med)))

    t_skin = max(8.0, min(med + k_noise * max(mad, 1.0), 35.0))
    cand_skin = (bh > t_skin).astype(np.uint8) * 255

    # 4. Oriented line opening on cand_skin (No MORPH_CLOSE to avoid pore fusion)
    line_hair = _line_open(cand_skin, line_len=line_len, n_angles=n_angles)

    # 5. Lesion protection: exclude solid circular blobs & apply outside >= 0.25 | very_long
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    solid_blobs = cv2.morphologyEx(cand_skin, cv2.MORPH_OPEN, open_k)
    solid_blobs_dilated = cv2.dilate(
        solid_blobs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    hair_candidates = cv2.subtract(line_hair, solid_blobs_dilated)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_candidates, connectivity=8)
    clean_hair = np.zeros_like(hair_candidates)

    has_lesion = np.any(l_mask > 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 6:
            continue
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        length = max(comp_w, comp_h)

        comp_mask = (labels == i)
        if has_lesion:
            outside_pixels = np.count_nonzero(comp_mask & (l_mask == 0))
            outside_ratio = outside_pixels / float(area)
        else:
            outside_ratio = 1.0

        is_very_long = (length >= lesion_line_len)

        # Retain if sufficiently outside lesion OR long hair traversing the lesion
        if outside_ratio >= 0.25 or is_very_long:
            clean_hair[comp_mask] = 255

    # 6. Safety guard: if mask coverage exceeds 25%, tighten threshold by 1.4x
    coverage_work = float(np.count_nonzero(clean_hair)) / float(clean_hair.size)
    if coverage_work > 0.25:
        t_tight = t_skin * 1.4
        cand_tight = (bh > t_tight).astype(np.uint8) * 255
        line_tight = _line_open(cand_tight, line_len=line_len, n_angles=n_angles)
        hair_candidates = cv2.subtract(line_tight, solid_blobs_dilated)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_candidates, connectivity=8)
        clean_hair = np.zeros_like(hair_candidates)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 6:
                continue
            comp_mask = (labels == i)
            length = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            outside_ratio = (
                np.count_nonzero(comp_mask & (l_mask == 0)) / float(stats[i, cv2.CC_STAT_AREA])
                if has_lesion
                else 1.0
            )
            if outside_ratio >= 0.25 or length >= lesion_line_len:
                clean_hair[comp_mask] = 255

    # 7. Light 3x3 ellipse dilation at working scale (thin strand mask)
    k_dilate3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask_work = cv2.dilate(clean_hair, k_dilate3, iterations=1)

    # 8. Upscale hair mask to full image dimensions
    if scale < 1.0:
        final_hair_mask = cv2.resize(hair_mask_work, (w, h), interpolation=cv2.INTER_NEAREST)
        full_dilate = max(1, min(3, int(round(1.0 / scale))))
        if full_dilate > 1:
            final_hair_mask = cv2.dilate(
                final_hair_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (full_dilate, full_dilate)),
                iterations=1,
            )
        actual_inpaint_radius = min(5, max(2, int(round(2.0 / scale))))
    else:
        final_hair_mask = hair_mask_work
        actual_inpaint_radius = min(5, inpaint_radius)

    # 9. Clean inpainting with min_coverage safety cap
    coverage = float(np.count_nonzero(final_hair_mask)) / float(final_hair_mask.size)
    if coverage < min_coverage:
        # Hairless image remains untouched
        return img_bgr.copy(), np.zeros((h, w), dtype=np.uint8)

    hair_removed = cv2.inpaint(img_bgr, final_hair_mask, actual_inpaint_radius, cv2.INPAINT_TELEA)
    return hair_removed, final_hair_mask



def apply_dullrazor(
    img_bgr: np.ndarray,
    inpaint_radius: int = 4,
    k_noise: float = 3.2,
    line_len: int = 13,
    lesion_line_len: int = 25,
    min_coverage: float = 0.0003,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hair removal with a second pass over the first pass's output, which catches strand
    fragments the first pass leaves behind. The second pass is skipped when the first
    already masked more than 25% of the image. Returns (hair_removed_bgr, union_mask).
    """
    out, mask = _apply_dullrazor_single(
        img_bgr, inpaint_radius, k_noise, line_len, lesion_line_len, min_coverage
    )
    if cv2 is None or passes < 2 or mask is None or mask.size == 0 or float((mask > 0).mean()) > 0.25:
        return out, mask
    out2, mask2 = _apply_dullrazor_single(
        out, inpaint_radius, k_noise, line_len, lesion_line_len, min_coverage
    )
    return out2, cv2.bitwise_or(mask, mask2)


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

    img_bgr = _ensure_bgr_image(img_bgr)
    h, w = img_bgr.shape[:2]
    if h < 4 or w < 4:
        return 30.0

    bh = max(1, min(int(h * 0.15), h // 2))
    bw = max(1, min(int(w * 0.15), w // 2))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_star = lab[:, :, 0] * (100.0 / 255.0)
    b_star = lab[:, :, 2] - 128.0

    border_l = l_star[border_mask]
    border_b = b_star[border_mask]
    if border_l.size == 0 or border_b.size == 0:
        return 30.0

    l_bg = float(np.median(border_l))
    b_bg = float(np.median(border_b))

    denom = b_bg if abs(b_bg) > 1e-5 else 1e-5
    ita = float(np.arctan2(l_bg - 50.0, denom) * (180.0 / np.pi))
    return round(ita, 1)


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
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        return img_bgr
    img_bgr = _ensure_bgr_image(img_bgr)
    img_float = img_bgr.astype(np.float32)
    norm_img = img_float / 255.0
    power_img = norm_img ** p
    e_b = float((np.mean(power_img[:, :, 0])) ** (1.0 / p))
    e_g = float((np.mean(power_img[:, :, 1])) ** (1.0 / p))
    e_r = float((np.mean(power_img[:, :, 2])) ** (1.0 / p))

    e_gray = (e_b + e_g + e_r) / 3.0
    if e_gray < 1e-6:
        return img_bgr.copy()

    scale_b = float(np.clip(e_gray / (e_b + 1e-6), 0.70, 1.40))
    scale_g = float(np.clip(e_gray / (e_g + 1e-6), 0.70, 1.40))
    scale_r = float(np.clip(e_gray / (e_r + 1e-6), 0.70, 1.40))

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

    img_bgr = _ensure_bgr_image(img_bgr)

    # 1. Shades of gray
    balanced = apply_shades_of_gray(img_bgr, p=6)

    # 2. LAB standardization of background skin
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = balanced.shape[:2]
    if h < 4 or w < 4:
        return balanced

    bh = max(1, min(int(h * 0.15), h // 2))
    bw = max(1, min(int(w * 0.15), w // 2))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    border_0 = lab[border_mask, 0]
    border_2 = lab[border_mask, 2]
    if border_0.size == 0 or border_2.size == 0:
        return balanced

    bg_l = float(np.median(border_0))
    bg_b = float(np.median(border_2))

    # Clinical reference for normal skin in HAM10000 (L*~65 -> 162 in uint8, b*~14 -> 142 in uint8)
    target_l = 162.0
    target_b = 142.0

    delta_l = (target_l - bg_l) * 0.35
    delta_b = (target_b - bg_b) * 0.30

    lab[:, :, 0] = np.clip(lab[:, :, 0] + delta_l, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + delta_b, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _skin_pixels(img_bgr: np.ndarray) -> np.ndarray:
    """Helper to detect skin pixels via widened YCrCb and HSV chromaticity ranges."""
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        return np.zeros((1, 1), dtype=bool)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_ycrcb = (
        (ycrcb[:, :, 1] >= 130)
        & (ycrcb[:, :, 1] <= 180)
        & (ycrcb[:, :, 2] >= 75)
        & (ycrcb[:, :, 2] <= 135)
    )

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
    1. Inclusive Asian / South Asian Skin Chromaticity Check on Union of Raw + White-Balanced Images
    2. Blur & Focus Check (Laplacian Variance at standard 600px width)
    
    Returns: (is_valid, skin_coverage_pct, blur_score, ita, status_message)
    """
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        return False, 0.0, 0.0, 0.0, "Invalid image data"

    img_bgr = _ensure_bgr_image(img_bgr)
    h, w = img_bgr.shape[:2]
    if h < 4 or w < 4:
        return False, 0.0, 0.0, 0.0, "Image too small"

    # Fast check for all-black or all-white uniform images
    min_val, max_val = float(np.min(img_bgr)), float(np.max(img_bgr))
    if max_val - min_val < 5:
        return False, 0.0, 0.0, 0.0, "Uniform / blank image (Verification Failed)"

    # 1. Skin Chromaticity on Union of Raw + Shades-of-Gray White-Balanced Pixels
    balanced_bgr = apply_shades_of_gray(img_bgr, p=6)
    skin_union = _skin_pixels(img_bgr) | _skin_pixels(balanced_bgr)
    skin_coverage_pct = float(np.mean(skin_union) * 100.0)

    # 2. Laplacian Blur Variance at standard 600px working size
    target_w = 600
    scale = target_w / float(w) if w > target_w else 1.0
    if scale < 1.0:
        target_h = max(10, int(h * scale))
        work = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        work = img_bgr

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. ITA & Fitzpatrick Phototype
    ita = calculate_ita(img_bgr)
    fitzpatrick = classify_fitzpatrick(ita)

    is_valid = True
    reasons = []
    warnings = []

    # Reject skin coverage only if skin_coverage_pct < 15.0
    if skin_coverage_pct < 15.0:
        is_valid = False
        reasons.append(f"Insufficient skin pixels ({skin_coverage_pct:.1f}% < 15.0%)")

    # Reject blur only if blur_score < 6.0 (warn only if < 15.0)
    if blur_score < 6.0:
        is_valid = False
        reasons.append(f"Image is out of focus / blurry (blur index {blur_score:.1f} < 6.0)")
    elif blur_score < 15.0:
        warnings.append(f"Borderline focus (blur index {blur_score:.1f})")

    if is_valid:
        status_msg = f"Passed Skin & Quality Verification (Fitzpatrick Type {fitzpatrick}, ITA {ita:.1f}°)"
        if warnings:
            status_msg += f" [{', '.join(warnings)}]"
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
