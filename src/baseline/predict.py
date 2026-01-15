# predict.py  (minimal safe edits)
import cv2
import os
import glob
import argparse
import numpy as np
import joblib
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd

# relative from src/baseline -> up two -> ai_trace_finder/models
MODEL_DIR = "../../models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RF_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "svm.pkl")

def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_metadata_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h if h != 0 else 0
    file_size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0

    pixels = img.flatten()
    mean_intensity = float(np.mean(pixels))
    std_intensity = float(np.std(pixels))
    skewness = float(skew(pixels))
    kurt = float(kurtosis(pixels))
    counts = np.histogram(pixels, bins=256, range=(0, 1))[0].astype(np.float64) + 1e-6
    ent = float(entropy(counts))

    edges = sobel(img)
    edge_density = float(np.mean(edges > 0.1))

    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def find_first_image(data_root="../../data"):
    # find common tif/jpg/png images under data_root (first match)
    patterns = ["**/*.tif", "**/*.tiff", "**/*.jpg", "**/*.jpeg", "**/*.png"]
    for p in patterns:
        matches = glob.glob(os.path.join(data_root, p), recursive=True)
        if matches:
            return matches[0]
    return None

def predict_scanner(img_path, model_choice="rf"):
    # check model files
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}")
    model_path = RF_PATH if model_choice == "rf" else SVM_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(model_path)

    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)

    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    pred = model.predict(X_scaled)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0]
    else:
        # optional fallback to decision_function -> softmax-like
        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_scaled)
                exps = np.exp(scores - np.max(scores))
                probs = (exps / np.sum(exps)).ravel()
                prob = probs
            except Exception:
                prob = None

    return pred, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", help="Path to image", default=None)
    parser.add_argument("--model", "-m", choices=["rf", "svm"], default="rf", help="Model choice")
    args = parser.parse_args()

    test_image = args.image
    if not test_image:
        test_image = find_first_image("../../data")
        if test_image:
            print(f"No image provided; using first found image: {test_image}")
    else:
        print(f"Using test image: {test_image}")

    if not test_image or not os.path.exists(test_image):
        raise FileNotFoundError(f"Test image not found at: {test_image}")

    pred, prob = predict_scanner(test_image, model_choice=args.model)
    print("Predicted Scanner:", pred)
    print("Class Probabilities:", prob)
    print("Prediction completed successfully!")
