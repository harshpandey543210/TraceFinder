"""
processing.py
Data loading and preprocessing functions.
- Handles `flatfield`, `official`, `wikipedia` datasets (your folder names).
- Produces: residual images (256x256x1) and saves them as pickle files.
"""

import os
import sys
import pickle
from tqdm import tqdm

# Make sure current folder (where utils.py lives) is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import process_batch_gpu, USE_GPU

# Fallback CPU
import cv2
import numpy as np
import pywt
from scipy.signal import wiener as scipy_wiener
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global Parameters
IMG_SIZE = (256, 256)
DENOISE_METHOD = "wavelet"
BATCH_SIZE = 64
MAX_WORKERS = 8

print(f"Dataset Processing: Use GPU? {USE_GPU}")


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_img(img):
    return img.astype(np.float32) / 255.0


def denoise_wavelet_img_cpu(img):
    """Wavelet denoising (Haar) on CPU using PyWavelets."""
    coeffs = pywt.dwt2(img, "haar")
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0
    cV[:] = 0
    cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), "haar")


def preprocess_image_cpu(fpath, method=DENOISE_METHOD):
    """Process single image on CPU."""
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = to_gray(img)
    img = resize_to(img)
    img = normalize_img(img)
    # Denoising
    if method == "wiener":
        den = scipy_wiener(img, mysize=(5, 5))
    elif method == "wavelet":
        den = denoise_wavelet_img_cpu(img)
    else:
        raise ValueError(f"Unknown denoise method: {method}")

    # Residual
    res = (img - den).astype(np.float32)
    return res


def process_folder(base_dir, use_dpi_subfolders=True):
    """
    Process all images under a folder.
    Returns nested dict: residuals[scanner][dpi] = list_of_residuals
    """
    residuals_dict = {}
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist!")
        return {}

    scanners = sorted(os.listdir(base_dir))

    for scanner in tqdm(scanners, desc="Scanners"):
        scanner_dir = os.path.join(base_dir, scanner)
        if not os.path.isdir(scanner_dir):
            continue

        residuals_dict[scanner] = {}

        if use_dpi_subfolders:
            # Your structure: base/scanner/<scanner_name>/<dpi>/images
            subfolders = [
                d
                for d in os.listdir(scanner_dir)
                if os.path.isdir(os.path.join(scanner_dir, d))
            ]
            if len(subfolders) == 1:
                inner_scanner_dir = os.path.join(scanner_dir, subfolders[0])
            else:
                inner_scanner_dir = scanner_dir  # fallback

            for dpi in os.listdir(inner_scanner_dir):
                dpi_dir = os.path.join(inner_scanner_dir, dpi)
                if not os.path.isdir(dpi_dir):
                    continue
                files = [
                    os.path.join(dpi_dir, f)
                    for f in os.listdir(dpi_dir)
                    if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
                ]

                # Process files
                residuals_dict[scanner][dpi] = run_processing(files)
        else:
            # Flatfield dataset: no dpi subfolders
            files = [
                os.path.join(scanner_dir, f)
                for f in os.listdir(scanner_dir)
                if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
            ]
            residuals_dict[scanner] = run_processing(files)

    return residuals_dict


def run_processing(file_list):
    """Dispatches to GPU batch or CPU pool."""
    if not file_list:
        return []

    all_residuals = []

    if USE_GPU:
        # Batch chunks
        for i in range(0, len(file_list), BATCH_SIZE):
            chunk = file_list[i : i + BATCH_SIZE]
            res_chunk = process_batch_gpu(chunk)
            all_residuals.extend(res_chunk)
    else:
        # Fallback to ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(preprocess_image_cpu, f) for f in file_list]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    all_residuals.append(res)

    return all_residuals


def save_pickle(obj, path):
    # Ensure dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved residuals to {path}")


# =================== MAIN EXECUTION (PATHS ADAPTED TO TraceFinder) ===================

if __name__ == "__main__":
    # Project root : TRACEFINDER/
    # Package root : ai_trace_finder/
    # Data folders : ai_trace_finder/data/{flatfield, official, wikipedia}
    # Results out  : ai_trace_finder/results/hybrid_cnn/*.pkl

    # ai_trace_finder/src  -> PROJECT_ROOT
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Base dir for datasets (ai_trace_finder/data)
    BASE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data"))

    # Official + Wikipedia
    datasets = ["official", "wikipedia"]
    official_wiki_residuals = {}

    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        dataset_dir = os.path.join(BASE_DIR, dataset)
        official_wiki_residuals[dataset] = process_folder(
            dataset_dir, use_dpi_subfolders=True
        )

    # Save under ai_trace_finder/results/hybrid_cnn/
    RESULTS_DIR = os.path.abspath(
        os.path.join(PROJECT_ROOT, "..", "results", "hybrid_cnn")
    )
    OUT_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
    save_pickle(official_wiki_residuals, OUT_PATH)

    # Flatfield dataset (no dpi subfolders)
    print("\nProcessing flatfield dataset...")
    flatfield_dir = os.path.join(BASE_DIR, "flatfield")
    flatfield_residuals = process_folder(flatfield_dir, use_dpi_subfolders=False)

    OUT_PATH_FLAT = os.path.join(RESULTS_DIR, "flatfield_residuals.pkl")
    save_pickle(flatfield_residuals, OUT_PATH_FLAT)

    # Summary (flatfield only)
    total_scanners = len(flatfield_residuals)
    total_images = sum(
        [len(v) for v in flatfield_residuals.values() if isinstance(v, list)]
    )
    print(f"\nDone. Flatfield: {total_scanners} scanners, ~{total_images} images processed.")
