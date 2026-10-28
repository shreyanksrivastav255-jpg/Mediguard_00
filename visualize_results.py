# visualize_results.py
# Stage 8: Visualization
#
# Generates and saves graphs comparing SmallCNN vs ResNet18 under
# each disturbance type, plus an overall robustness score comparison.
# Saves PNGs into a results/ folder for use in the project report/demo.
#
# Run with:
#   python visualize_results.py

import os
import torch
import matplotlib.pyplot as plt
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
RESULTS_DIR = "results"


def build_resnet18(num_classes=2):
    model = tv_models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_accuracy(adapter, loader):
    y_true, y_pred = adapter.predict(loader)
    return accuracy_score(y_true, y_pred)


def collect_curve(adapter, loader_fn, param_dict_or_list, is_dict):
    """Returns (labels, accuracies) for a disturbance sweep."""
    labels = []
    accs = []
    if is_dict:
        for label, param in param_dict_or_list.items():
            if isinstance(param, tuple):
                loader, _ = loader_fn(*param)
            else:
                loader, _ = loader_fn(param)
            labels.append(label)
            accs.append(get_accuracy(adapter, loader))
    else:
        for severity in param_dict_or_list:
            loader, _ = loader_fn(severity)
            labels.append(str(severity))
            accs.append(get_accuracy(adapter, loader))
    return labels, accs


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, _, clean_test_loader, class_names = get_dataloaders()

    print("Loading models...")
    model_a = SmallCNN(num_classes=2).to(device)
    model_a.load_state_dict(torch.load("pneumonia_cnn_best.pt", map_location=device))
    adapter_a = ModelAdapter(model_a, device, name="SmallCNN")

    model_b = build_resnet18(num_classes=2).to(device)
    model_b.load_state_dict(torch.load("pneumonia_resnet18_best.pt", map_location=device))
    adapter_b = ModelAdapter(model_b, device, name="ResNet18")

    clean_acc_a = get_accuracy(adapter_a, clean_test_loader)
    clean_acc_b = get_accuracy(adapter_b, clean_test_loader)

    disturbances = {
        "Noise": (get_noisy_test_loader, NOISE_SEVERITIES, False),
        "Blur": (get_blurred_test_loader, BLUR_SEVERITIES, True),
        "Brightness": (get_brightness_test_loader, BRIGHTNESS_SEVERITIES, True),
        "Contrast": (get_contrast_test_loader, CONTRAST_SEVERITIES, True),
    }

    category_avg_a = {}
    category_avg_b = {}

    # --- Per-disturbance line charts ---
    for name, (loader_fn, params, is_dict) in disturbances.items():
        print(f"Evaluating {name}...")
        labels_a, accs_a = collect_curve(adapter_a, loader_fn, params, is_dict)
        labels_b, accs_b = collect_curve(adapter_b, loader_fn, params, is_dict)

        category_avg_a[name] = sum(accs_a) / len(accs_a)
        category_avg_b[name] = sum(accs_b) / len(accs_b)

        plt.figure(figsize=(7, 5))
        x = range(len(labels_a))
        plt.plot(x, accs_a, marker='o', label='SmallCNN')
        plt.plot(x, accs_b, marker='s', label='ResNet18')
        plt.axhline(y=clean_acc_a, color='gray', linestyle='--', alpha=0.5, label='SmallCNN clean')
        plt.axhline(y=clean_acc_b, color='lightgray', linestyle='--', alpha=0.5, label='ResNet18 clean')
        plt.xticks(list(x), labels_a, rotation=30, ha='right')
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs {name} Severity")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"accuracy_vs_{name.lower()}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved {out_path}")

    # --- Robustness score bar chart ---
    def compute_score(clean_acc, category_avg):
        retentions = [min(acc / clean_acc, 1.0) for acc in category_avg.values()]
        return (sum(retentions) / len(retentions)) * 100

    score_a = compute_score(clean_acc_a, category_avg_a)
    score_b = compute_score(clean_acc_b, category_avg_b)

    plt.figure(figsize=(6, 5))
    plt.bar(["SmallCNN", "ResNet18"], [score_a, score_b], color=["#4C72B0", "#DD8452"])
    plt.ylabel("Robustness Score (0-100)")
    plt.title("Overall Robustness Score Comparison")
    plt.ylim(0, 100)
    for i, v in enumerate([score_a, score_b]):
        plt.text(i, v + 1, f"{v:.1f}", ha='center')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "robustness_score_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # --- Category comparison grouped bar chart ---
    categories = list(category_avg_a.keys())
    a_vals = [category_avg_a[c] for c in categories]
    b_vals = [category_avg_b[c] for c in categories]

    x = range(len(categories))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], a_vals, width, label='SmallCNN')
    plt.bar([i + width/2 for i in x], b_vals, width, label='ResNet18')
    plt.xticks(list(x), categories)
    plt.ylabel("Average Accuracy")
    plt.title("Average Accuracy per Disturbance Category")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "category_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    print("\nAll graphs saved in the 'results/' folder.")


if __name__ == "__main__":
    main()