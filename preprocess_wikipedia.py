import os
import cv2
import csv
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy


DATASET_OFFICIAL = "ai_trace_finder\data\wikipedia"
OUTPUT_DIR = "ai_trace_finder\processed_data\wikipedia"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata_features.csv")


def load_and_preprocess(img_path, size=(512, 512)):
    img = io.imread(img_path, as_gray=True)
    img = img_as_float(img)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# Noise residual
def extract_noise_residual(img):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised

# Patch extraction
def extract_patches(img, patch_size=128, stride=128):
    patches = []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
    return patches

# Metadata computation
def compute_metadata_features(img, file_path, scanner_id, resolution="unknown"):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "file_name": os.path.basename(file_path),
        "main_class": "Official",
        "resolution": resolution,
        "class_label": scanner_id,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

# Main preprocessing function
def preprocess_official_dataset(official_dir, out_dir, csv_path):
    fieldnames = [
        "file_name", "main_class", "resolution", "class_label",
        "width", "height", "aspect_ratio", "file_size_kb",
        "mean_intensity", "std_intensity", "skewness", "kurtosis",
        "entropy", "edge_density"
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Walk through all folders and subfolders
        for root, dirs, files in os.walk(official_dir):
            # Skip root if it's the main dataset folder (to infer scanner_id)
            if root == official_dir:
                continue

            # Infer scanner_id from first level folder
            scanner_id = os.path.basename(os.path.dirname(root))
            subfolder_name = os.path.basename(root)

            save_path = os.path.join(out_dir, scanner_id, subfolder_name)
            os.makedirs(save_path, exist_ok=True)

            files_processed = 0
            for file in files:
                if not file.lower().endswith(('.png', '.tif', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(root, file)
                try:
                    img = load_and_preprocess(img_path)
                except Exception as e:
                    print(f" Failed to read {img_path}: {e}")
                    continue

                residual = extract_noise_residual(img)
                patches = extract_patches(residual)
                for idx, patch in enumerate(patches):
                    out_name = f"{os.path.splitext(file)[0]}_{idx}.npy"
                    np.save(os.path.join(save_path, out_name), patch)

                features = compute_metadata_features(img, img_path, scanner_id)
                writer.writerow(features)
                files_processed += 1

            if files_processed > 0:
                print(f" Processed {files_processed} files in folder: {root}")

if __name__ == "__main__":
    preprocess_official_dataset(DATASET_OFFICIAL, OUTPUT_DIR, CSV_PATH)
    print(" Official preprocessing + metadata feature extraction complete.")