# robustness_score.py
# Stage 6: Aggregates results across all 4 disturbance types (noise,
# blur, brightness, contrast) into a single Robustness Score (0-100)
# and a human-readable rating, per the Mediguard methodology.
#
# METHODOLOGY (documented explicitly so it can be justified in the report):
#
# For each disturbance category:
#   1. Run the model on the test set at several severities of that
#      disturbance.
#   2. Compute accuracy at each severity.
#   3. Average those accuracies -> "category accuracy".
#   4. Compute "retention" = category accuracy / clean accuracy,
#      capped at 1.0 (a disturbance that happens to slightly improve
#      accuracy shouldn't inflate the score above what clean gives).
#
# Overall Robustness Score = mean(retention across all 4 categories) * 100
#
# This rewards a model that keeps performing close to its own clean
# baseline under real-world-like distortions, regardless of how good
# the clean baseline itself is - i.e. it measures STABILITY, not raw
# accuracy. A model with 70% clean accuracy that barely drops under
# disturbance scores well; a model with 95% clean accuracy that
# collapses to 20% under mild noise scores poorly. This matches the
# project's core idea: robustness, not just accuracy.
#
# Rating bands (from the project synopsis, provisional - can be
# revisited later with more models/literature support):
#   90-100 -> Excellent
#   75-89  -> Good
#   60-74  -> Moderate
#   40-59  -> Weak
#   <40    -> Poor
#
# Run with:
#   python robustness_score.py

import torch
from sklearn.metrics import accuracy_score

from dataset_loader import get_dataloaders
from train_cnn import SmallCNN
from evaluate_baseline import get_all_predictions

from evaluate_noise import get_noisy_test_loader
from evaluate_blur import get_blurred_test_loader, BLUR_SEVERITIES
from evaluate_brightness import get_brightness_test_loader, BRIGHTNESS_SEVERITIES
from evaluate_contrast import get_contrast_test_loader, CONTRAST_SEVERITIES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep in sync with evaluate_noise.py's severities list
NOISE_SEVERITIES = [0.05, 0.10, 0.20]


def get_accuracy(model, loader):
    y_true, y_pred = get_all_predictions(model, loader)
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


def main():
    model = SmallCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("pneumonia_cnn_best.pt", map_location=device))

    # --- Clean baseline ---
    _, _, clean_test_loader, class_names = get_dataloaders()
    clean_acc = get_accuracy(model, clean_test_loader)
    print(f"Clean accuracy: {clean_acc:.4f}\n")

    category_results = {}

    # --- Noise ---
    print("Evaluating Noise...")
    noise_accs = []
    for severity in NOISE_SEVERITIES:
        loader, _ = get_noisy_test_loader(severity)
        acc = get_accuracy(model, loader)
        noise_accs.append(acc)
        print(f"  severity {severity}: acc={acc:.4f}")
    category_results["Noise"] = sum(noise_accs) / len(noise_accs)

    # --- Blur ---
    print("Evaluating Blur...")
    blur_accs = []
    for severity, (kernel_size, sigma) in BLUR_SEVERITIES.items():
        loader, _ = get_blurred_test_loader(kernel_size, sigma)
        acc = get_accuracy(model, loader)
        blur_accs.append(acc)
        print(f"  severity {severity}: acc={acc:.4f}")
    category_results["Blur"] = sum(blur_accs) / len(blur_accs)

    # --- Brightness ---
    print("Evaluating Brightness...")
    brightness_accs = []
    for label, factor in BRIGHTNESS_SEVERITIES.items():
        loader, _ = get_brightness_test_loader(factor)
        acc = get_accuracy(model, loader)
        brightness_accs.append(acc)
        print(f"  {label} (factor={factor}): acc={acc:.4f}")
    category_results["Brightness"] = sum(brightness_accs) / len(brightness_accs)

    # --- Contrast ---
    print("Evaluating Contrast...")
    contrast_accs = []
    for label, factor in CONTRAST_SEVERITIES.items():
        loader, _ = get_contrast_test_loader(factor)
        acc = get_accuracy(model, loader)
        contrast_accs.append(acc)
        print(f"  {label} (factor={factor}): acc={acc:.4f}")
    category_results["Contrast"] = sum(contrast_accs) / len(contrast_accs)

    # --- Compute retention + final score ---
    print("\n=== Category Summary ===")
    print(f"{'Category':<12} {'Avg Accuracy':<14} {'Retention vs Clean':<20}")
    retentions = []
    for category, avg_acc in category_results.items():
        retention = min(avg_acc / clean_acc, 1.0)
        retentions.append(retention)
        print(f"{category:<12} {avg_acc:<14.4f} {retention:<20.4f}")

    robustness_score = (sum(retentions) / len(retentions)) * 100
    rating = rating_for_score(robustness_score)

    print("\n=== FINAL ROBUSTNESS REPORT ===")
    print(f"Clean Accuracy:      {clean_acc:.4f}")
    print(f"Robustness Score:    {robustness_score:.2f} / 100")
    print(f"Rating:              {rating}")


if __name__ == "__main__":
    main()