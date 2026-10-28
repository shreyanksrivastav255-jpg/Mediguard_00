# dataset_loader.py
# Loads the Kaggle chest X-ray pneumonia dataset (train/val/test)
# using PyTorch's ImageFolder, with basic preprocessing.
#
# Folder structure expected:
#   data/chest_xray/chest_xray/train/NORMAL
#   data/chest_xray/chest_xray/train/PNEUMONIA
#   data/chest_xray/chest_xray/val/NORMAL
#   data/chest_xray/chest_xray/val/PNEUMONIA
#   data/chest_xray/chest_xray/test/NORMAL
#   data/chest_xray/chest_xray/test/PNEUMONIA

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---- Configuration ----
DATA_ROOT = "data/chest_xray/chest_xray"  # relative to project root
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # no worker processes; avoids Windows crash

# ---- Transforms ----
# Chest X-rays are grayscale but stored as RGB jpgs in this dataset.
# We resize, convert to tensor, and normalize using ImageNet stats
# (standard practice even for grayscale-ish medical images fed as RGB).
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),  # mild augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def get_dataloaders():
    """Returns train_loader, val_loader, test_loader, class_names"""

    train_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}/train", transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}/val", transform=eval_transforms
    )
    test_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}/test", transform=eval_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    class_names = train_dataset.classes  # ['NORMAL', 'PNEUMONIA']

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check when run directly:
    # python dataset_loader.py
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print("Classes:", class_names)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Grab one batch to confirm shapes
    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)  # [batch, 3, 224, 224]
    print("Batch label shape:", labels.shape)
    print("Sample labels:", labels[:10])