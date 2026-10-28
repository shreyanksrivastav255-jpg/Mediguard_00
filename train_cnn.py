# train_cnn.py (Mediguard's own training script - separate from Yale's train.py)
# Trains a small CNN on the chest X-ray pneumonia dataset and saves it.
#
# Run with:
#   python train_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_dataloaders

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Small CNN model ----
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, criterion):
    """Returns average loss and accuracy on the given loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print("Classes:", class_names)

    model = SmallCNN(num_classes=len(class_names)).to(device)

    # Class imbalance handling: this dataset has more PNEUMONIA than NORMAL
    # images in train, so we weight the loss accordingly.
    # (Approx counts: NORMAL ~1341, PNEUMONIA ~3875 in train set)
    class_weights = torch.tensor([3875 / 1341, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "pneumonia_cnn_best.pt")
            print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

    # Final test evaluation using the best saved model
    model.load_state_dict(torch.load("pneumonia_cnn_best.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinal Test | loss: {test_loss:.4f} acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()