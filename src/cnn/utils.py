"""
TraceFinder Utils - Feature Extraction & GPU Processing
Supports Hybrid CNN training pipeline with PRNU correlation, FFT, LBP, and texture features.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import local_binary_pattern as sk_lbp
from scipy.fft import fft2, fftshift
from scipy import ndimage
from typing import List, Dict, Any, Tuple, Union

# ------------------------------------------------------------------
# Module Setup
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# GPU Detection (singleton)
gpus = tf.config.list_physical_devices('GPU')
USE_GPU = len(gpus) > 0
print(f"Hybrid utils: GPU available? {USE_GPU}")

# ------------------------------------------------------------------
# Core Correlation Functions
# ------------------------------------------------------------------

def corr2d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute 2D normalized cross-correlation (CPU version).
    
    Args:
        a, b: 2D arrays (H, W) or residuals
        
    Returns:
        float: Correlation coefficient [-1, 1]
    """
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

def batch_corr_gpu(
    residuals: Union[List[np.ndarray], np.ndarray],
    fingerprints: Dict[str, np.ndarray],
    fp_keys: List[str]
) -> np.ndarray:
    """
    GPU-accelerated batch correlation for multiple residuals vs fingerprints.
    
    Args:
        residuals: List/array of shape (N, H, W) or (N, H, W, 1)
        fingerprints: Dict {scanner_name: fingerprint_array}
        fp_keys: Ordered list of scanner names
        
    Returns:
        np.ndarray: Shape (N, K) correlation matrix
    """
    if not residuals:
        return np.array([])

    # Normalize fingerprints -> (D, K) where D = H*W, K = num_scanners
    fp_matrix = []
    for k in fp_keys:
        fp = fingerprints[k].astype(np.float32).ravel()
        fp -= fp.mean()
        norm = np.linalg.norm(fp)
        fp_matrix.append(fp / (norm + 1e-8))

    fp_matrix = np.array(fp_matrix).T  # (D, K)

    # Normalize residuals -> (N, D)
    res_matrix = []
    for r in residuals:
        r_flat = r.astype(np.float32).ravel()
        r_flat -= r_flat.mean()
        norm = np.linalg.norm(r_flat)
        res_matrix.append(r_flat / (norm + 1e-8))

    res_matrix = np.array(res_matrix)  # (N, D)

    # GPU Matrix Multiplication: (N, D) @ (D, K) -> (N, K)
    with tf.device('/GPU:0' if USE_GPU else '/CPU:0'):
        A = tf.convert_to_tensor(res_matrix, dtype=tf.float32)
        B = tf.convert_to_tensor(fp_matrix, dtype=tf.float32)
        C = tf.matmul(A, B)

    return C.numpy()

# ------------------------------------------------------------------
# Enhanced Feature Extractors
# ------------------------------------------------------------------

def fft_radial_energy(img: np.ndarray, K: int = 6) -> List[float]:
    """Radial FFT energy spectrum (K radial bins)."""
    f = fftshift(fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K + 1)
    feats = [
        float(np.mean(mag[(r >= bins[i]) & (r < bins[i + 1])]))
        for i in range(K)
    ]
    return feats

def lbp_hist_safe(img: np.ndarray, P: int = 8, R: float = 1.0) -> List[float]:
    """Safe LBP histogram computation with normalization."""
    rng = float(np.ptp(img))
    if rng < 1e-12:
        return [0.0] * (P + 2)
    
    g = (img - img.min()) / (rng + 1e-8)
    g8 = (g * 255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32).tolist()

def extract_enhanced_features(residual: np.ndarray) -> List[float]:
    """
    Extract multi-scale texture features from residual:
    - FFT frequency bands (3)
    - LBP histogram (26 bins)
    - Gradient texture stats (4)
    
    Returns:
        List[float]: 33-dimensional feature vector
    """
    # FFT Frequency Bands
    f = fftshift(fft2(residual))
    mag = np.abs(f)
    h, w = mag.shape
    center_h, center_w = h // 2, w // 2

    low_freq = np.mean(mag[max(0, center_h - 20):center_h + 20, 
                          max(0, center_w - 20):center_w + 20])
    mid_region = mag[max(0, center_h - 60):center_h + 60, 
                     max(0, center_w - 60):center_w + 60]
    mid_freq = np.mean(mid_region) - low_freq
    high_freq = np.mean(mag) - np.mean(mid_region)

    # LBP Histogram (24 neighbors, uniform)
    lbp = sk_lbp(residual, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)

    # Gradient Texture Features
    grad_x = ndimage.sobel(residual, axis=1)
    grad_y = ndimage.sobel(residual, axis=0)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    texture_features = [
        np.std(residual),
        np.mean(np.abs(residual)),
        np.std(grad_mag),
        np.mean(grad_mag),
    ]

    return [float(low_freq), float(mid_freq), float(high_freq)] + \
           lbp_hist.tolist() + texture_features

# ------------------------------------------------------------------
# GPU Image Preprocessing Pipeline
# ------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize preserving aspect ratio."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] float32."""
    return img.astype(np.float32) / 255.0

def process_batch_gpu(file_paths: List[str]) -> List[np.ndarray]:
    """
    GPU-accelerated batch preprocessing + residual extraction.
    Approximates Haar wavelet denoising (L1 approximation).
    """
    imgs = []
    valid_indices = []

    # Load & Preprocess
    for idx, fpath in enumerate(file_paths):
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = to_gray(img)
        img = resize_to(img)
        img = normalize_img(img)
        imgs.append(img)
        valid_indices.append(idx)

    if not imgs:
        return []

    # GPU Batch Processing
    batch_np = np.array(imgs, dtype=np.float32)[..., np.newaxis]  # (N, 256, 256, 1)
    
    with tf.device('/GPU:0' if USE_GPU else '/CPU:0'):
        x = tf.convert_to_tensor(batch_np)

        # Haar L1 Approximation (downsample + upsample)
        pooled = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='VALID')
        denoised = tf.image.resize(
            pooled, (256, 256), 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        residual = x - denoised  # Noise residual

    return [residual.numpy()[i].squeeze() for i in range(len(residual))]
