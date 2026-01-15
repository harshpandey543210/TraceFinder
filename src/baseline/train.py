import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Paths (updated to match project structure; script runs from src/baseline)
CSV_PATH = "../../processed_data/metadata_features_for_training.csv"
MODEL_DIR = "../../models/baseline"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_baseline_models():
    print(f"Loading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found. Run preprocessing first.")
        return

    if df.empty:
        print("Error: Dataset is empty.")
        return

    print(f"Dataset loaded. Shape: {df.shape}")
    # safe check if dataset_source exists (may not in your CSV)
    if "dataset_source" in df.columns:
        print(f"Sources: {df['dataset_source'].unique()}")

    # Features and Target
    # Dropping non-feature columns
    drop_cols = ["file_name", "dataset_source", "main_class", "resolution", "class_label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df["class_label"]

    print(f"Features: {feature_cols}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Encoding Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    if len(le.classes_) < 2:
        print(f"Error: Not enough classes for classification. Found {len(le.classes_)}: {le.classes_}")
        print("Wait for preprocessing to finish extracting data from multiple classes.")
        return

    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # Train/Test Split
    # Stratify by target to ensure balanced classes in split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    # --- Random Forest ---
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print("Classification Report (RF):")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))

    # --- SVM ---
    print("\n--- Training SVM ---")
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {acc_svm:.4f}")
    print("Classification Report (SVM):")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_))
    
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm.joblib"))

if __name__ == "__main__":
    train_baseline_models()
