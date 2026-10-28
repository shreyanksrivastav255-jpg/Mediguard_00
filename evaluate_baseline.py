# evaluate_baseline.py
# Loads the trained model and reports full baseline metrics on the
# clean (undisturbed) test set: accuracy, precision, recall, F1,
# confusion matrix, and sensitivity/specificity.
#
# This is the "before disturbance" reference point that later
# disturbance experiments will be compared against.
#
# Run with:
#   python evaluate_baseline.py

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from dataset_loader import get_dataloaders
from train_cnn import SmallCNN  # reuse the same model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_predictions(model, loader):
    """Runs the model over a full loader and returns (y_true, y_pred) as lists."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    return y_true, y_pred


def print_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    # class index 1 = PNEUMONIA (the "positive" class we care about clinically)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)  # = sensitivity
    f1 = f1_score(y_true, y_pred, pos_label=1)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision (PNEUM.):  {precision:.4f}")
    print(f"Recall / Sensitivity:{recall:.4f}")
    print(f"Specificity:         {specificity:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"                Predicted NORMAL   Predicted PNEUMONIA")
    print(f"Actual NORMAL        {tn:>6}              {fp:>6}")
    print(f"Actual PNEUMONIA     {fn:>6}              {tp:>6}")
    print()
    print("False Positives (healthy misdiagnosed as pneumonia):", fp)
    print("False Negatives (pneumonia missed / undiagnosed):    ", fn)
    print()
    print("Full classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    model = SmallCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("pneumonia_cnn_best.pt", map_location=device))

    print("=== Baseline evaluation on CLEAN test set ===\n")
    y_true, y_pred = get_all_predictions(model, test_loader)
    metrics = print_metrics(y_true, y_pred, class_names)

    return metrics


if __name__ == "__main__":
    main()