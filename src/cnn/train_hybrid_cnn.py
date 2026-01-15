import os
import sys
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------------------
# Ensure imports work from src/cnn
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import corr2d, extract_enhanced_features
from modelcnn import build_hybrid_model  # <- use modelcnn.py

# ------------------------------------------------------------------
# Paths adapted to TraceFinder layout
# Project root : .../TraceFinder/
# Package root : .../TraceFinder/ai_trace_finder/
# Results in   : ai_trace_finder/results/hybrid_cnn/...
# ------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "results", "hybrid_cnn"))

RES_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
FP_PATH = os.path.join(RESULTS_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(RESULTS_DIR, "fp_keys.npy")
FEATURES_PATH = os.path.join(RESULTS_DIR, "features.pkl")
ENHANCED_PATH = os.path.join(RESULTS_DIR, "enhanced_features.pkl")

ART_DIR = RESULTS_DIR  # store hybrid artifacts together
os.makedirs(ART_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------------
# GPU Setup
# ------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
device_name = "/GPU:0" if gpus else "/CPU:0"
print("Using device:", device_name)

# ------------------------------------------------------------------
# Load & Prepare Data
# ------------------------------------------------------------------
print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

# Optional: Load pre-computed features
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
    print("Pre-computed features not found. Computing on the fly (this may be slow)...")
    if os.path.exists(FP_PATH):
        with open(FP_PATH, "rb") as f:
            scanner_fps = pickle.load(f)
        fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
    else:
        raise FileNotFoundError("Scanner fingerprints not found. Run feature_extrac.py first.")

X_img, X_feat, y = [], [], []
idx_counter = 0

for dataset_name in ["official", "wikipedia"]:
    if dataset_name not in residuals_dict:
        continue

    print(f"Processing {dataset_name}...")
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        if isinstance(dpi_dict, dict):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    # Image Input (256, 256, 1)
                    X_img.append(np.expand_dims(res, -1))

                    # Feature Input
                    if precomputed:
                        if idx_counter >= len(feats_prnu) or idx_counter >= len(feats_enh):
                            continue
                        f_p = feats_prnu[idx_counter]
                        f_e = feats_enh[idx_counter]
                        X_feat.append(f_p + f_e)
                    else:
                        v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                        v_enh = extract_enhanced_features(res)
                        X_feat.append(v_corr + v_enh)

                    y.append(scanner)
                    idx_counter += 1

X_img = np.array(X_img, dtype=np.float32)
X_feat = np.array(X_feat, dtype=np.float32)
y = np.array(y)

print(f"Dataset Shape: Images {X_img.shape}, Features {X_feat.shape}, Labels {y.shape}")

if len(X_img) == 0 or len(X_feat) == 0:
    raise RuntimeError(
        "No samples assembled for training. Check features.pkl/enhanced_features.pkl and residuals_dict."
    )

# ------------------------------------------------------------------
# Encode Labels & Split
# ------------------------------------------------------------------
le = LabelEncoder()
y_int = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = to_categorical(y_int, num_classes)

X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
    X_img, X_feat, y_cat, test_size=0.2, random_state=SEED, stratify=y_int
)

# Normalize Features
scaler = StandardScaler()
X_feat_tr = scaler.fit_transform(X_feat_tr)
X_feat_te = scaler.transform(X_feat_te)

# Save Preprocessors
with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ------------------------------------------------------------------
# Build & Train Model
# ------------------------------------------------------------------
with tf.device(device_name):
    model = build_hybrid_model(
        img_shape=(256, 256, 1),
        feat_shape=(X_feat.shape[1],),
        num_classes=num_classes,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Data Pipeline
    BATCH = 32
    train_ds = (
        tf.data.Dataset.from_tensor_slices(((X_img_tr, X_feat_tr), y_tr))
        .shuffle(len(y_tr))
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(((X_img_te, X_feat_te), y_te))
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Callbacks
    ckpt_path = os.path.join(ART_DIR, "scanner_hybrid.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-6, monitor="val_accuracy"
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True, monitor="val_accuracy"
        ),
    ]

    # Train
    history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)

    # Save Final Model & History
    model.save(os.path.join(ART_DIR, "scanner_hybrid_final.keras"))
    with open(os.path.join(ART_DIR, "hybrid_training_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

# ------------------------------------------------------------------
# Plot Training Results
# ------------------------------------------------------------------
def plot_history(hist, save_dir):
    acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training acc")
    plt.plot(epochs, val_acc, "r*-", label="Validation acc")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "r*-", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "training_plot.png")
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")


plot_history(history, ART_DIR)
print("Training complete!")
