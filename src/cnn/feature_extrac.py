import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

# Ensure current folder (where utils.py lives) is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import batch_corr_gpu, extract_enhanced_features

# ------------------------------------------------------------------
# Paths adapted to TraceFinder layout
# Project root : .../TraceFinder/
# Package root : .../TraceFinder/ai_trace_finder/
# Data         : ai_trace_finder/data/...
# Results      : ai_trace_finder/results/hybrid_cnn/...
# ------------------------------------------------------------------

# ai_trace_finder/src  -> SRC_DIR
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "results", "hybrid_cnn"))

FLATFIELD_RESIDUALS_PATH = os.path.join(RESULTS_DIR, "flatfield_residuals.pkl")
FP_OUT_PATH = os.path.join(RESULTS_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(RESULTS_DIR, "fp_keys.npy")

RES_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
FEATURES_OUT = os.path.join(RESULTS_DIR, "features.pkl")
ENHANCED_OUT = os.path.join(RESULTS_DIR, "enhanced_features.pkl")

# ============================================================
# 1) Compute scanner fingerprints from flatfield residuals
# ============================================================
if os.path.exists(FLATFIELD_RESIDUALS_PATH):
    with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
        flatfield_residuals = pickle.load(f)

    scanner_fingerprints = {}
    print("Computing fingerprints from Flatfields...")
    for scanner, residuals in flatfield_residuals.items():
        if not residuals:
            continue
        stack = np.stack(residuals, axis=0)      # (N, H, W)
        fingerprint = np.mean(stack, axis=0)     # (H, W)
        scanner_fingerprints[scanner] = fingerprint

    # Save fingerprints
    os.makedirs(os.path.dirname(FP_OUT_PATH), exist_ok=True)
    with open(FP_OUT_PATH, "wb") as f:
        pickle.dump(scanner_fingerprints, f)

    # Save deterministic scanner order
    fp_keys = sorted(scanner_fingerprints.keys())
    np.save(ORDER_NPY, np.array(fp_keys))
    print(f"Saved {len(scanner_fingerprints)} fingerprints and fp_keys.npy")

else:
    print(f"Warning: {FLATFIELD_RESIDUALS_PATH} not found. Skipping fingerprint generation.")
    if os.path.exists(FP_OUT_PATH):
        with open(FP_OUT_PATH, "rb") as f:
            scanner_fingerprints = pickle.load(f)
        if os.path.exists(ORDER_NPY):
            fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
        else:
            fp_keys = sorted(scanner_fingerprints.keys())
    else:
        print("No fingerprints found. Cannot proceed with PRNU features.")
        exit(1)

# ============================================================
# 2) PRNU Features (Batch GPU)
# ============================================================
if os.path.exists(RES_PATH):
    # Load residuals
    with open(RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)

    features, labels = [], []

    # Note: processing keys as stored in residuals_dict
    for dataset_name in ["official", "wikipedia"]:
        if dataset_name not in residuals_dict:
            print(f"Dataset {dataset_name} not in residuals.")
            continue

        print(f"Computing PRNU features for {dataset_name} (GPU Batch) ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            if isinstance(dpi_dict, dict):
                for dpi, res_list in dpi_dict.items():
                    if not res_list:
                        continue
                    corrs = batch_corr_gpu(res_list, scanner_fingerprints, fp_keys)  # (N, K)
                    features.extend(corrs.tolist())
                    labels.extend([scanner] * len(res_list))

    # Save features
    with open(FEATURES_OUT, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)

    if len(features) > 0:
        print(f"Saved features shape: {len(features)} x {len(features[0])}")
    else:
        print("Saved features.pkl but feature list is empty (0 samples).")

    # ========================================================
    # 3) Enhanced Features (FFT + LBP + Texture)
    # ========================================================
    enhanced_features, enhanced_labels = [], []
    for dataset_name in ["official", "wikipedia"]:
        if dataset_name not in residuals_dict:
            continue

        print(f"Extracting enhanced features for {dataset_name} ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            if isinstance(dpi_dict, dict):
                for dpi, res_list in dpi_dict.items():
                    for res in res_list:
                        feat = extract_enhanced_features(res)
                        enhanced_features.append(feat)
                        enhanced_labels.append(scanner)

    # Save enhanced features
    with open(ENHANCED_OUT, "wb") as f:
        pickle.dump({"features": enhanced_features, "labels": enhanced_labels}, f)

    if len(enhanced_features) > 0:
        print(
            f"Enhanced features shape: {len(enhanced_features)} x {len(enhanced_features[0])}"
        )
    else:
        print("Saved enhanced_features.pkl but list is empty (0 samples).")

    print(f"Saved enhanced features to {ENHANCED_OUT}")

else:
    print(f"{RES_PATH} not found. Cannot extract features.")
