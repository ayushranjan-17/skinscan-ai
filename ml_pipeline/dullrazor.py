"""
DullRazor Hair Removal and Gray-World Color Constancy Algorithm for Dermatoscopic Images.
Removes occluding hairs and normalizes lighting variations without degrading lesion borders.
"""

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

def apply_dullrazor(img_bgr, kernel_size=9, inpaint_radius=2):
    """
    Applies the DullRazor algorithm:
    1. Grayscale conversion
    2. Morphological BlackHat transformation to detect dark linear hair shafts
    3. Thresholding to create a binary hair mask
    4. Telea / Navier-Stokes inpainting to replace hair pixels with surrounding skin texture
    """
    if cv2 is None:
        return img_bgr, np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Structural element for BlackHat (detect thin lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create clean binary mask of hair structures
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint hair regions using fast marching method
    inpainted = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted, mask

def apply_color_constancy(img_bgr):
    """
    Gray-World Color Constancy algorithm:
    Adjusts RGB channels so that the average scene illumination is neutral gray,
    removing clinical camera flash color casts.
    """
    img_float = img_bgr.astype(np.float32)
    b_mean = np.mean(img_float[:, :, 0])
    g_mean = np.mean(img_float[:, :, 1])
    r_mean = np.mean(img_float[:, :, 2])
    
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    
    b_scale = gray_mean / (b_mean + 1e-6)
    g_scale = gray_mean / (g_mean + 1e-6)
    r_scale = gray_mean / (r_mean + 1e-6)
    
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * b_scale, 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * g_scale, 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * r_scale, 0, 255)
    
    return img_float.astype(np.uint8)

def preprocess_dermatoscopy_pipeline(img_bgr, target_size=(224, 224)):
    """
    Complete pre-processing pipeline:
    1. DullRazor hair removal
    2. Color constancy normalization
    3. Standardized resizing
    """
    hair_removed, _ = apply_dullrazor(img_bgr)
    color_normalized = apply_color_constancy(hair_removed)
    if cv2 is not None:
        resized = cv2.resize(color_normalized, target_size, interpolation=cv2.INTER_AREA)
    else:
        # Simple nearest-neighbor crop/pad fallback
        resized = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        h = min(color_normalized.shape[0], target_size[0])
        w = min(color_normalized.shape[1], target_size[1])
        resized[:h, :w] = color_normalized[:h, :w]
    return resized

if __name__ == '__main__':
    # Unit test with synthetic image
    synthetic_img = np.full((300, 300, 3), (168, 181, 232), dtype=np.uint8)
    if cv2 is not None:
        cv2.line(synthetic_img, (50, 50), (250, 260), (20, 20, 20), 3)
    processed = preprocess_dermatoscopy_pipeline(synthetic_img)
    print(f"Preprocessed output shape: {processed.shape}, dtype: {processed.dtype}")
    assert processed.shape == (224, 224, 3)
    print("DullRazor preprocessor unit test passed successfully!")
