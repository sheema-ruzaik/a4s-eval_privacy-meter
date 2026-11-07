import numpy as np
import torch

from a4s_eval.data_model.evaluation import ModelConfig
from a4s_eval.service.functional_model import ImageClassificationModel
from a4s_eval.typing import Array


def load_torch_image_classification(model_config: ModelConfig) -> ImageClassificationModel:
    """Load a PyTorch image classification model.

    Args:
        model_config: Configuration containing model path and metadata

    Returns:
        ImageClassificationModel with predict functions
    """
    model = torch.jit.load(model_config.path)
    model.eval()

    def predict_proba(x: Array) -> Array:
        """Predict class probabilities for image inputs.

        Args:
            x: Image array of shape (batch_size, channels, height, width) or (batch_size, height, width, channels)

        Returns:
            Array of probabilities for each class
        """
        if isinstance(x, np.ndarray):
            # Handle different input formats
            if x.ndim == 3:
                # Single image: (C, H, W) or (H, W, C)
                if x.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
                    x = np.transpose(x, (2, 0, 1))
                x = np.expand_dims(x, axis=0)  # Add batch dimension
            elif x.ndim == 4 and x.shape[-1] == 3:
                # Batch: (N, H, W, C) -> (N, C, H, W)
                x = np.transpose(x, (0, 3, 1, 2))

            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            logits = model(x)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict_class(x: Array) -> Array:
        """Predict class labels for image inputs.

        Args:
            x: Image array

        Returns:
            Array of predicted class indices
        """
        y_pred = predict_proba(x)
        return np.argmax(y_pred, axis=-1)

    def predict_proba_grad(x: Array) -> Array:
        """Predict probabilities with gradient computation enabled.

        Args:
            x: Image array

        Returns:
            Tensor of probabilities with gradient tracking
        """
        if isinstance(x, np.ndarray):
            # Handle different input formats
            if x.ndim == 3:
                if x.shape[-1] == 3:
                    x = np.transpose(x, (2, 0, 1))
                x = np.expand_dims(x, axis=0)
            elif x.ndim == 4 and x.shape[-1] == 3:
                x = np.transpose(x, (0, 3, 1, 2))

            x = torch.tensor(x, dtype=torch.float32)

        x = x.requires_grad_()
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    return ImageClassificationModel(
        predict_class=predict_class,
        predict_proba=predict_proba,
        predict_proba_grad=predict_proba_grad,
    )
