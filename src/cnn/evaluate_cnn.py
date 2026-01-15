import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# Correct project imports
from dataset import get_dataloaders
from model import SimpleCNN


def evaluate_model(
        model_path="ai_trace_finder/models/cnn/cnn_model.pth",
        batch_size=32,
        data_path="ai_trace_finder/data"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading test dataset...")
    _, _, test_loader, classes = get_dataloaders(
        data_root=data_path,
        batch_size=batch_size
    )
    num_classes = len(classes)
    
    # Load Model
    model = SimpleCNN(num_classes=num_classes).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # OPTIONAL: Enable if you want a heatmap saved.
    # fig_path = "ai_trace_finder/models/cnn/confusion_matrix.png"
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig(fig_path)
    # print(f"Confusion matrix saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN Model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="ai_trace_finder/models/cnn/cnn_model.pth",
        help="Path to saved model"
    )

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size
    )
