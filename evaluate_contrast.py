# evaluate_contrast.py
# Applies contrast variation disturbance to the test set images and
# re-evaluates the trained model, comparing against the clean baseline.
#
# Fourth and final core disturbance (Stage 4) - completes the
# noise / blur / brightness / contrast set from the synopsis.
#
# Like brightness, contrast is bidirectional: low contrast (washed out,
# flat-looking) and high contrast (harsh, overly sharp) can both occur
# in real-world imaging due to device/sensor differences.
#
# Applied to the PIL image before ToTensor/Normalize, same as blur/brightness.
#
# Run with:
#   python evaluate_contrast.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_loader import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from train_cnn import SmallCNN
from evaluate_baseline import get_all_predictions, print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ColorJitter contrast factor, applied as a FIXED value (factor, factor)
# for reproducibility, same approach as brightness.
#
# factor < 1.0 -> lower contrast (flatter/washed out)
# factor > 1.0 -> higher contrast (harsher)
CONTRAST_SEVERITIES = {
    "low_flat":    0.85,
    "medium_flat": 0.60,
    "high_flat":   0.35,
    "low_harsh":   1.15,
    "medium_harsh": 1.40,
    "high_harsh":  1.70,
}


def get_contrast_test_loader(factor):
    """Builds a test DataLoader with a fixed contrast factor applied."""
    contrast_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(contrast=(factor, factor)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    test_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}/test", transform=contrast_transforms
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )
    return test_loader, test_dataset.classes


def main():
    model = SmallCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("pneumonia_cnn_best.pt", map_location=device))

    results = {}

    for label, factor in CONTRAST_SEVERITIES.items():
        print(f"\n=== Contrast - {label} (factor={factor}) ===\n")
        loader, class_names = get_contrast_test_loader(factor)
        y_true, y_pred = get_all_predictions(model, loader)
        metrics = print_metrics(y_true, y_pred, class_names)
        results[label] = metrics

    print("\n\n=== SUMMARY: Accuracy vs Contrast ===")
    print(f"{'Condition':<16} {'Factor':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12}")
    for label, factor in CONTRAST_SEVERITIES.items():
        m = results[label]
        print(f"{label:<16} {factor:<8} {m['accuracy']:<10.4f} "
              f"{m['recall_sensitivity']:<12.4f} {m['specificity']:<12.4f}")


if __name__ == "__main__":
    main()