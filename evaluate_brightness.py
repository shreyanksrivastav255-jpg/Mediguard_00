# evaluate_brightness.py
# Applies brightness variation disturbance to the test set images and
# re-evaluates the trained model, comparing against the clean baseline.
#
# Third piece of the "Disturbance Engine" (Stage 4).
#
# Unlike noise/blur, brightness disturbance is bidirectional -
# real-world X-rays can be over- or under-exposed. We test both
# directions separately since a model may fail differently for each.
#
# Applied to the PIL image before ToTensor/Normalize, same as blur.
#
# Run with:
#   python evaluate_brightness.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_loader import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from train_cnn import SmallCNN
from evaluate_baseline import get_all_predictions, print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ColorJitter's brightness factor is a range [max(0, 1-b), 1+b] to sample from.
# We apply it as a FIXED shift (not random) per severity so results are
# reproducible and directly comparable, by using brightness=(factor, factor).
#
# factor < 1.0 -> darker (underexposed)
# factor > 1.0 -> brighter (overexposed)
BRIGHTNESS_SEVERITIES = {
    "low_dark":    0.85,
    "medium_dark": 0.65,
    "high_dark":   0.40,
    "low_bright":  1.15,
    "medium_bright": 1.35,
    "high_bright": 1.60,
}


def get_brightness_test_loader(factor):
    """Builds a test DataLoader with a fixed brightness factor applied."""
    brightness_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=(factor, factor)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    test_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}/test", transform=brightness_transforms
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

    for label, factor in BRIGHTNESS_SEVERITIES.items():
        print(f"\n=== Brightness - {label} (factor={factor}) ===\n")
        loader, class_names = get_brightness_test_loader(factor)
        y_true, y_pred = get_all_predictions(model, loader)
        metrics = print_metrics(y_true, y_pred, class_names)
        results[label] = metrics

    print("\n\n=== SUMMARY: Accuracy vs Brightness ===")
    print(f"{'Condition':<16} {'Factor':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12}")
    for label, factor in BRIGHTNESS_SEVERITIES.items():
        m = results[label]
        print(f"{label:<16} {factor:<8} {m['accuracy']:<10.4f} "
              f"{m['recall_sensitivity']:<12.4f} {m['specificity']:<12.4f}")


if __name__ == "__main__":
    main()