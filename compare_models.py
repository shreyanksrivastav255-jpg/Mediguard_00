# compare_models.py
# Stage 7: Multiple Models
#
# Runs the full robustness evaluation pipeline (clean baseline + all
# 4 disturbances + robustness score) on MULTIPLE models via the
# ModelAdapter, and prints a side-by-side comparison.
#
# This is the actual "FSSAI for ML models" moment: two different,
# independently-trained models both get rated on the same standard,
# so you can directly compare how trustworthy each one is under
# real-world image degradation - not just which one is "more accurate".
#
# Run with:
#   python compare_models.py

import torch
from sklearn.metrics import accuracy_score
from torchvision import models as tv_models
import torch.nn as nn

from dataset_loader import get_dataloaders
from train_cnn import SmallCNN
from model_adapter import ModelAdapter

from evaluate_noise import get_noisy_test_loader
from evaluate_blur import get_blurred_test_loader, BLUR_SEVERITIES
from evaluate_brightness import get_brightness_test_loader, BRIGHTNESS_SEVERITIES
from evaluate_contrast import get_contrast_test_loader, CONTRAST_SEVERITIES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE_SEVERITIES = [0.05, 0.10, 0.20]


def build_resnet18(num_classes=2):
    model = tv_models.resnet18(weights=None)  # weights loaded from our checkpoint, not ImageNet again
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_accuracy(adapter, loader):
    y_true, y_pred = adapter.predict(loader)
    return accuracy_score(y_true, y_pred)


def rating_for_score(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Moderate"
    elif score >= 40:
        return "Weak"
    else:
        return "Poor"


def evaluate_robustness(adapter, clean_test_loader):
    """Runs one model (via its adapter) through the full robustness pipeline.
    Returns a dict of results."""

    clean_acc = get_accuracy(adapter, clean_test_loader)

    category_results = {}

    noise_accs = []
    for severity in NOISE_SEVERITIES:
        loader, _ = get_noisy_test_loader(severity)
        noise_accs.append(get_accuracy(adapter, loader))
    category_results["Noise"] = sum(noise_accs) / len(noise_accs)

    blur_accs = []
    for severity, (kernel_size, sigma) in BLUR_SEVERITIES.items():
        loader, _ = get_blurred_test_loader(kernel_size, sigma)
        blur_accs.append(get_accuracy(adapter, loader))
    category_results["Blur"] = sum(blur_accs) / len(blur_accs)

    brightness_accs = []
    for label, factor in BRIGHTNESS_SEVERITIES.items():
        loader, _ = get_brightness_test_loader(factor)
        brightness_accs.append(get_accuracy(adapter, loader))
    category_results["Brightness"] = sum(brightness_accs) / len(brightness_accs)

    contrast_accs = []
    for label, factor in CONTRAST_SEVERITIES.items():
        loader, _ = get_contrast_test_loader(factor)
        contrast_accs.append(get_accuracy(adapter, loader))
    category_results["Contrast"] = sum(contrast_accs) / len(contrast_accs)

    retentions = [min(acc / clean_acc, 1.0) for acc in category_results.values()]
    robustness_score = (sum(retentions) / len(retentions)) * 100
    rating = rating_for_score(robustness_score)

    return {
        "clean_accuracy": clean_acc,
        "category_accuracies": category_results,
        "robustness_score": robustness_score,
        "rating": rating,
    }


def main():
    _, _, clean_test_loader, class_names = get_dataloaders()

    # --- Model 1: SmallCNN (trained from scratch) ---
    print("Loading Model A: SmallCNN...")
    model_a = SmallCNN(num_classes=2).to(device)
    model_a.load_state_dict(torch.load("pneumonia_cnn_best.pt", map_location=device))
    adapter_a = ModelAdapter(model_a, device, name="SmallCNN")

    # --- Model 2: ResNet18 (pretrained + fine-tuned) ---
    print("Loading Model B: ResNet18...")
    model_b = build_resnet18(num_classes=2).to(device)
    model_b.load_state_dict(torch.load("pneumonia_resnet18_best.pt", map_location=device))
    adapter_b = ModelAdapter(model_b, device, name="ResNet18")

    print("\nEvaluating Model A (SmallCNN)...")
    results_a = evaluate_robustness(adapter_a, clean_test_loader)

    print("Evaluating Model B (ResNet18)...")
    results_b = evaluate_robustness(adapter_b, clean_test_loader)

    # --- Side-by-side comparison ---
    print("\n\n=== MODEL COMPARISON REPORT ===\n")
    print(f"{'Metric':<20} {'SmallCNN':<15} {'ResNet18':<15}")
    print(f"{'Clean Accuracy':<20} {results_a['clean_accuracy']:<15.4f} {results_b['clean_accuracy']:<15.4f}")
    for category in results_a["category_accuracies"]:
        acc_a = results_a["category_accuracies"][category]
        acc_b = results_b["category_accuracies"][category]
        print(f"{category + ' Avg Acc':<20} {acc_a:<15.4f} {acc_b:<15.4f}")
    print(f"{'Robustness Score':<20} {results_a['robustness_score']:<15.2f} {results_b['robustness_score']:<15.2f}")
    print(f"{'Rating':<20} {results_a['rating']:<15} {results_b['rating']:<15}")


if __name__ == "__main__":
    main()