import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import corr2d, extract_enhanced_features

# ------------------------------------------------------------------
# Paths adapted to TraceFinder
# ------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "results", "hybrid_cnn"))

MODEL_PATH   = os.path.join(RESULTS_DIR, "scanner_hybrid_final.keras")
ENCODER_PATH = os.path.join(RESULTS_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH  = os.path.join(RESULTS_DIR, "hybrid_feat_scaler.pkl")
RES_PATH     = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
FP_PATH      = os.path.join(RESULTS_DIR, "scanner_fingerprints.pkl")
ORDER_NPY    = os.path.join(RESULTS_DIR, "fp_keys.npy")
FEATURES_PATH  = os.path.join(RESULTS_DIR, "features.pkl")
ENHANCED_PATH  = os.path.join(RESULTS_DIR, "enhanced_features.pkl")

ART_DIR = RESULTS_DIR  # save confusion matrix here as well
os.makedirs(ART_DIR, exist_ok=True)

# ---- Reproducibility ----
SEED = 42
np.random.seed(SEED)

# ---- Load artifacts ----
print(f"Loading model from {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---- Load residuals ----
print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

# ---- Load pre-computed features if available ----
precomputed = False
if os.path.exists(FEATURES_PATH) and os.path.exists(ENHANCED_PATH):
    print("Loading pre-computed features...")
    with open(FEATURES_PATH, "rb") as f:
        d_feat = pickle.load(f)
        feats_prnu = d_feat["features"]
    with open(ENHANCED_PATH, "rb") as f:
        d_enh = pickle.load(f)
        feats_enh = d_enh["features"]
    precomputed = True
else:
    print("Pre-computed features not found. Computing on the fly...")
    if os.path.exists(FP_PATH):
        with open(FP_PATH, "rb") as f:
            scanner_fps = pickle.load(f)
        fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
    else:
        raise FileNotFoundError("Scanner fingerprints not found. Run feature_extrac.py first.")

# ---- Reconstruct full dataset from residuals ----
X_img_all, X_feat_all, y_all = [], [], []
idx_counter = 0

for dataset_name in ["official", "wikipedia"]:
    if dataset_name not in residuals_dict:
        continue

    print(f"Processing {dataset_name}...")
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        if isinstance(dpi_dict, dict):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    X_img_all.append(np.expand_dims(res, -1))

                    if precomputed:
                        if idx_counter >= len(feats_prnu) or idx_counter >= len(feats_enh):
                            continue
                        f_p = feats_prnu[idx_counter]
                        f_e = feats_enh[idx_counter]
                        X_feat_all.append(f_p + f_e)
                    else:
                        v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                        v_enh = extract_enhanced_features(res)
                        X_feat_all.append(v_corr + v_enh)

                    y_all.append(scanner)
                    idx_counter += 1

X_img_all = np.array(X_img_all, dtype=np.float32)
X_feat_all = np.array(X_feat_all, dtype=np.float32)
y_all = np.array(y_all)

print(f"Full dataset: Images {X_img_all.shape}, Features {X_feat_all.shape}, Labels {y_all.shape}")

# ---- Split to recover Test Set (same split as training) ----
y_int_all = le.transform(y_all)
num_classes = len(le.classes_)
y_cat_all = to_categorical(y_int_all, num_classes)

print("Splitting data to recover Test Set...")
_, X_img_te, _, X_feat_te, _, y_te = train_test_split(
    X_img_all,
    X_feat_all,
    y_cat_all,
    test_size=0.2,
    random_state=SEED,
    stratify=y_int_all,
)

y_int_te = np.argmax(y_te, axis=1)

# Scale Features
X_feat_te = scaler.transform(X_feat_te)

print(f"Test Set: {len(X_img_te)} samples")

# ---- Evaluate ----
print("Running prediction...")
y_pred_prob = model.predict([X_img_te, X_feat_te])
y_pred = np.argmax(y_pred_prob, axis=1)

test_acc = accuracy_score(y_int_te, y_pred)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_int_te, y_pred, target_names=le.classes_))

# ---- Confusion Matrix ----
cm = confusion_matrix(y_int_te, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Acc: {test_acc*100:.2f}%)")

cm_path = os.path.join(ART_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Saved confusion matrix to {cm_path}")

