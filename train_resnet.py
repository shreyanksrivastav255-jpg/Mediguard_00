    # train_resnet.py
# Trains a SECOND model for Mediguard to test: a pretrained ResNet18
# (ImageNet weights) fine-tuned on the pneumonia dataset.
#
# This is architecturally different from SmallCNN (our first model)
# and uses transfer learning - much closer to how real deployed
# medical models are typically built (like Yale's VGG16 approach).
#
# Run with:
#   python train_resnet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset_loader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def build_resnet18(num_classes=2):
    """Loads ImageNet-pretrained ResNet18 and replaces the final layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, criterion):
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

    model = build_resnet18(num_classes=len(class_names)).to(device)

    # Same class imbalance handling as train_cnn.py
    class_weights = torch.tensor([3875 / 1341, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Lower learning rate than train_cnn.py - fine-tuning a pretrained
    # model needs smaller updates so we don't destroy the useful
    # ImageNet features it already learned.
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 5  # fewer epochs needed - pretrained weights converge faster
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
            torch.save(model.state_dict(), "pneumonia_resnet18_best.pt")
            print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

    model.load_state_dict(torch.load("pneumonia_resnet18_best.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinal Test | loss: {test_loss:.4f} acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()