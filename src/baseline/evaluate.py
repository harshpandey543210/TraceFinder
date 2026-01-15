import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Relative path from src/baseline -> up two -> processed_data
CSV_PATH = "../../processed_data/metadata_features_for_training.csv"

# Relative paths for models/scaler (will resolve to ai_trace_finder/models)
SCALER_PATH = "../../models/scaler.pkl"

def evaluate_model(model_path, name, save_dir="../../results"):
    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # Drop only the target column — keep all other columns as features
    X = df.drop(columns=["class_label"])
    y = df["class_label"]

    # Load scaler + model
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(model_path)

    # Transform features
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Print report
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y, y_pred))

    # Determine labels for confusion matrix (fallback to unique y if model has no classes_)
    labels = getattr(model, "classes_", np.unique(y))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues"
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Ensure results directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

if __name__ == "__main__":
    evaluate_model("../../models/random_forest.pkl", "Random Forest")
    evaluate_model("../../models/svm.pkl", "SVM")
