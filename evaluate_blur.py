# evaluate_blur.py
# Applies Gaussian blur disturbance to the test set images and
# re-evaluates the trained model, comparing against the clean baseline.
#
# Second piece of the "Disturbance Engine" (Stage 4).
#
# Note: unlike noise, blur is applied to the PIL image BEFORE
# ToTensor/Normalize - blurring is a spatial smoothing operation
# and should happen on the raw image, not on normalized tensor values.
#
# Run with:
#   python evaluate_blur.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset_loader import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from train_cnn import SmallCNN
from evaluate_baseline import get_all_predictions, print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Severity -> (kernel_size, sigma)
# kernel_size must be odd. Larger kernel + higher sigma = more blur.
BLUR_SEVERITIES = {
    0.05: (3, 1.0),   # low
    0.10: (7, 2.5),   # medium
    0.20: (15, 5.0),  # high
}


def get_blurred_test_loader(kernel_size, sigma):
    """Builds a test DataLoader with Gaussian blur applied at the given strength."""
    blurred_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    test_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}/test", transform=blurred_transforms
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

    for severity, (kernel_size, sigma) in BLUR_SEVERITIES.items():
        print(f"\n=== Gaussian Blur - severity {severity} (kernel={kernel_size}, sigma={sigma}) ===\n")
        blurred_loader, class_names = get_blurred_test_loader(kernel_size, sigma)
        y_true, y_pred = get_all_predictions(model, blurred_loader)
        metrics = print_metrics(y_true, y_pred, class_names)
        results[severity] = metrics

    print("\n\n=== SUMMARY: Accuracy vs Blur Severity ===")
    print(f"{'Severity':<10} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12}")
    for severity, m in results.items():
        print(f"{severity:<10} {m['accuracy']:<10.4f} "
              f"{m['recall_sensitivity']:<12.4f} {m['specificity']:<12.4f}")


if __name__ == "__main__":
    main()