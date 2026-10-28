# robustness_report.py
# Generic single-model robustness evaluation.
# Works with ANY model registered in MODEL_REGISTRY below - just
# pick which one to test, no need for a separate script per model.
#
# Run with:
#   python robustness_report.py
# (it will prompt you to choose a model)
#
# Or specify directly:
#   python robustness_report.py smallcnn
#   python robustness_report.py resnet18

import sys
import torch
import torch.nn as nn
from torchvision import models as tv_models
from sklearn.metrics import accuracy_score

from dataset_loader import get_dataloaders
from train_cnn import SmallCNN
from model_adapter import ModelAdapter
from evaluate_baseline import get_all_predictions, print_metrics

from evaluate_noise import get_noisy_test_loader
from evaluate_blur import get_blurred_test_loader, BLUR_SEVERITIES
from evaluate_brightness import get_brightness_test_loader, BRIGHTNESS_SEVERITIES
from evaluate_contrast import get_contrast_test_loader, CONTRAST_SEVERITIES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE_SEVERITIES = [0.05, 0.10, 0.20]


def build_smallcnn(num_classes=2):
    return SmallCNN(num_classes=num_classes)


def build_resnet18(num_classes=2):
    model = tv_models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# --- MODEL_REGISTRY ---
# To add a new model later: write a build_xxx() function above that
# returns an uninitialized nn.Module with the right output size, then
# add an entry here pointing to its saved checkpoint file.
MODEL_REGISTRY = {
    "smallcnn": {
        "display_name": "SmallCNN (trained from scratch)",
        "build_fn": build_smallcnn,
        "checkpoint": "pneumonia_cnn_best.pt",
    },
    "resnet18": {
        "display_name": "ResNet18 (pretrained + fine-tuned)",
        "build_fn": build_resnet18,
        "checkpoint": "pneumonia_resnet18_best.pt",
    },
}


def choose_model():
    """Returns the model key, either from CLI arg or interactive prompt."""
    if len(sys.argv) > 1:
        key = sys.argv[1].lower()
        if key not in MODEL_REGISTRY:
            print(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        return key

    print("Available models:")
    keys = list(MODEL_REGISTRY.keys())
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {MODEL_REGISTRY[key]['display_name']}")
    choice = input(f"Choose a model (1-{len(keys)}): ").strip()
    try:
        idx = int(choice) - 1
        return keys[idx]
    except (ValueError, IndexError):
        print("Invalid choice.")
        sys.exit(1)


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


def get_accuracy(adapter, loader):
    y_true, y_pred = adapter.predict(loader)
    return accuracy_score(y_true, y_pred)


def main():
    model_key = choose_model()
    config = MODEL_REGISTRY[model_key]
    print(f"\n=== Robustness Report: {config['display_name']} ===\n")

    model = config["build_fn"](num_classes=2).to(device)
    model.load_state_dict(torch.load(config["checkpoint"], map_location=device))
    adapter = ModelAdapter(model, device, name=model_key)

    _, _, clean_test_loader, class_names = get_dataloaders()

    print("--- Clean Baseline ---")
    y_true, y_pred = adapter.predict(clean_test_loader)
    clean_metrics = print_metrics(y_true, y_pred, class_names)
    clean_acc = clean_metrics["accuracy"]

    category_avgs = {}

    print("\n--- Noise ---")
    accs = []
    for severity in NOISE_SEVERITIES:
        loader, _ = get_noisy_test_loader(severity)
        acc = get_accuracy(adapter, loader)
        accs.append(acc)
        print(f"  severity {severity}: acc={acc:.4f}")
    category_avgs["Noise"] = sum(accs) / len(accs)

    print("\n--- Blur ---")
    accs = []
    for severity, (k, s) in BLUR_SEVERITIES.items():
        loader, _ = get_blurred_test_loader(k, s)
        acc = get_accuracy(adapter, loader)
        accs.append(acc)
        print(f"  severity {severity}: acc={acc:.4f}")
    category_avgs["Blur"] = sum(accs) / len(accs)

    print("\n--- Brightness ---")
    accs = []
    for label, factor in BRIGHTNESS_SEVERITIES.items():
        loader, _ = get_brightness_test_loader(factor)
        acc = get_accuracy(adapter, loader)
        accs.append(acc)
        print(f"  {label} (factor={factor}): acc={acc:.4f}")
    category_avgs["Brightness"] = sum(accs) / len(accs)

    print("\n--- Contrast ---")
    accs = []
    for label, factor in CONTRAST_SEVERITIES.items():
        loader, _ = get_contrast_test_loader(factor)
        acc = get_accuracy(adapter, loader)
        accs.append(acc)
        print(f"  {label} (factor={factor}): acc={acc:.4f}")
    category_avgs["Contrast"] = sum(accs) / len(accs)

    print("\n=== Category Summary ===")
    print(f"{'Category':<12} {'Avg Accuracy':<14} {'Retention vs Clean':<20}")
    retentions = []
    for category, avg_acc in category_avgs.items():
        retention = min(avg_acc / clean_acc, 1.0)
        retentions.append(retention)
        print(f"{category:<12} {avg_acc:<14.4f} {retention:<20.4f}")

    robustness_score = (sum(retentions) / len(retentions)) * 100
    rating = rating_for_score(robustness_score)

    print(f"\n=== FINAL ROBUSTNESS REPORT: {config['display_name']} ===")
    print(f"Clean Accuracy:      {clean_acc:.4f}")
    print(f"Robustness Score:    {robustness_score:.2f} / 100")
    print(f"Rating:              {rating}")


if __name__ == "__main__":
    main()