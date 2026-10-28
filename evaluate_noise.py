# evaluate_noise.py
# Applies Gaussian noise disturbance to the test set images and
# re-evaluates the trained model, comparing against the clean baseline.
#
# This is the first working piece of the "Disturbance Engine" from
# the Mediguard architecture (Stage 4).
#
# Run with:
#   python evaluate_noise.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_loader import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from train_cnn import SmallCNN
from evaluate_baseline import get_all_predictions, print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization stats (must match what the model was trained with)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class AddGaussianNoise:
    """
    Adds Gaussian noise directly to a tensor image, AFTER normalization.
    severity controls the standard deviation of the noise.

    Rough guide (tune based on results):
      severity 0.05 -> low noise
      severity 0.10 -> medium noise
      severity 0.20 -> high noise
    """
    def __init__(self, severity=0.1):
        self.severity = severity

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.severity
        return tensor + noise


def get_noisy_test_loader(severity):
    """Builds a test DataLoader with Gaussian noise applied at the given severity."""
    noisy_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        AddGaussianNoise(severity=severity),
    ])

    test_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}/test", transform=noisy_transforms
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )
    return test_loader, test_dataset.classes


def main():
    model = SmallCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("pneumonia_cnn_best.pt", map_location=device))

    severities = [0.05, 0.10, 0.20]  # low, medium, high

    results = {}

    for severity in severities:
        print(f"\n=== Gaussian Noise - severity {severity} ===\n")
        noisy_loader, class_names = get_noisy_test_loader(severity)
        y_true, y_pred = get_all_predictions(model, noisy_loader)
        metrics = print_metrics(y_true, y_pred, class_names)
        results[severity] = metrics

    print("\n\n=== SUMMARY: Accuracy vs Noise Severity ===")
    print(f"{'Severity':<10} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12}")
    for severity, m in results.items():
        print(f"{severity:<10} {m['accuracy']:<10.4f} "
              f"{m['recall_sensitivity']:<12.4f} {m['specificity']:<12.4f}")


if __name__ == "__main__":
    main()