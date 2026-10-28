# model_adapter.py
# Stage 3: Model Adapter
#
# Wraps any already-trained PyTorch model behind a single uniform
# interface, so the rest of Mediguard (disturbance engine, evaluation,
# scoring) never needs to know what's inside the model - it just
# calls adapter.predict(loader) the same way every time.
#
# This is what makes "multiple models" possible: as long as a model
# can be wrapped in a ModelAdapter, Mediguard can test it.

import torch


class ModelAdapter:
    """
    Generic adapter around a PyTorch nn.Module.

    Assumes:
      - model outputs raw logits of shape [batch, num_classes]
      - predicted class = argmax over logits
      - model has already been loaded with trained weights
      - model is already moved to the correct device

    This covers the vast majority of PyTorch image classifiers
    (custom CNNs, torchvision models like ResNet/VGG, etc.) without
    modification. Future support for other frameworks (TensorFlow,
    ONNX) would mean adding new adapter classes with the same
    predict() interface, not changing any calling code.
    """

    def __init__(self, model, device, name="model"):
        self.model = model
        self.device = device
        self.name = name
        self.model.eval()

    def predict(self, loader):
        """
        Runs the model over a full DataLoader.
        Returns (y_true, y_pred) as plain Python lists.
        """
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu()
                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())
        return y_true, y_pred

    def __repr__(self):
        return f"ModelAdapter(name={self.name})"