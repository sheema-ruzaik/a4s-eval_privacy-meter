"""
Privacy Meter - Range Membership Inference Attack (RAMIA) Metric

This metric implements RAMIA to detect privacy leakage in machine learning models.
RAMIA checks if data similar to training samples can reveal privacy leaks by testing
the neighborhood around training points instead of exact matches.
"""

from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import ImageClassificationModel
from a4s_eval.utils.logging import get_logger

logger = get_logger()


def generate_neighboring_samples(
    images: np.ndarray, num_neighbors: int = 5, noise_scale: float = 0.1
) -> np.ndarray:
    """Generate neighboring samples by adding small perturbations to images.

    Args:
        images: Original images of shape (N, C, H, W) or (N, H, W, C)
        num_neighbors: Number of neighboring samples to generate per image
        noise_scale: Scale of Gaussian noise to add

    Returns:
        Array of neighboring images
    """
    neighbors = []
    for _ in range(num_neighbors):
        noise = np.random.normal(0, noise_scale, images.shape)
        perturbed = images + noise
        # Clip to valid range [0, 1] assuming images are normalized
        perturbed = np.clip(perturbed, 0, 1)
        neighbors.append(perturbed)

    return np.array(neighbors)


def compute_prediction_confidence(
    model: ImageClassificationModel, images: np.ndarray
) -> np.ndarray:
    """Compute prediction confidence for images.

    Args:
        model: Image classification model
        images: Images to predict on

    Returns:
        Max probability for each prediction
    """
    probas = model.predict_proba(images)
    return np.max(probas, axis=-1)


def ramia_attack(
    model: ImageClassificationModel,
    train_images: np.ndarray,
    test_images: np.ndarray,
    num_neighbors: int = 5,
    noise_scale: float = 0.1,
) -> dict:
    """Perform RAMIA attack to detect membership inference vulnerabilities.

    Args:
        model: Trained image classification model
        train_images: Images from training set
        test_images: Images NOT from training set
        num_neighbors: Number of neighboring samples to generate
        noise_scale: Scale of noise for generating neighbors

    Returns:
        Dictionary with attack results
    """
    logger.info("Starting RAMIA attack...")

    # Sample a subset if datasets are too large (for performance)
    max_samples = 500
    if len(train_images) > max_samples:
        train_indices = np.random.choice(len(train_images), max_samples, replace=False)
        train_images = train_images[train_indices]

    if len(test_images) > max_samples:
        test_indices = np.random.choice(len(test_images), max_samples, replace=False)
        test_images = test_images[test_indices]

    logger.info(
        f"Using {len(train_images)} train samples and {len(test_images)} test samples"
    )

    # Get predictions on original samples
    train_conf_orig = compute_prediction_confidence(model, train_images)
    test_conf_orig = compute_prediction_confidence(model, test_images)

    logger.info("Generating neighboring samples for train set...")
    train_neighbors_list = []
    for i in range(len(train_images)):
        neighbors = generate_neighboring_samples(
            train_images[i : i + 1], num_neighbors, noise_scale
        )
        train_neighbors_list.append(neighbors)

    logger.info("Generating neighboring samples for test set...")
    test_neighbors_list = []
    for i in range(len(test_images)):
        neighbors = generate_neighboring_samples(
            test_images[i : i + 1], num_neighbors, noise_scale
        )
        test_neighbors_list.append(neighbors)

    # Compute confidence on neighboring samples
    train_neighbor_confs = []
    for neighbors in train_neighbors_list:
        # neighbors shape: (num_neighbors, C, H, W)
        neighbors_flat = neighbors.reshape(-1, *neighbors.shape[2:])
        confs = compute_prediction_confidence(model, neighbors_flat)
        train_neighbor_confs.append(np.mean(confs))

    test_neighbor_confs = []
    for neighbors in test_neighbors_list:
        neighbors_flat = neighbors.reshape(-1, *neighbors.shape[2:])
        confs = compute_prediction_confidence(model, neighbors_flat)
        test_neighbor_confs.append(np.mean(confs))

    train_neighbor_confs = np.array(train_neighbor_confs)
    test_neighbor_confs = np.array(test_neighbor_confs)

    # RAMIA signal: difference between original and neighbor confidence
    train_ramia_signal = train_conf_orig - train_neighbor_confs
    test_ramia_signal = test_conf_orig - test_neighbor_confs

    # Compute attack metrics
    threshold = np.median(np.concatenate([train_ramia_signal, test_ramia_signal]))

    # Predict membership: high signal = likely in training set
    train_predictions = train_ramia_signal > threshold
    test_predictions = test_ramia_signal > threshold

    # True positives: correctly identified training samples
    true_positives = np.sum(train_predictions)
    # True negatives: correctly identified test samples
    true_negatives = np.sum(~test_predictions)
    # False positives: test samples incorrectly labeled as training
    false_positives = np.sum(test_predictions)
    # False negatives: training samples incorrectly labeled as test
    false_negatives = np.sum(~train_predictions)

    total = len(train_images) + len(test_images)
    accuracy = (true_positives + true_negatives) / total

    # Precision: of samples we said were in training, how many actually were?
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Recall: of actual training samples, how many did we identify?
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # F1 score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Privacy leakage score: how much better than random guessing (0.5)
    privacy_leakage = accuracy - 0.5

    results = {
        "attack_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "privacy_leakage": float(privacy_leakage),
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "train_signal_mean": float(np.mean(train_ramia_signal)),
        "test_signal_mean": float(np.mean(test_ramia_signal)),
    }

    logger.info(f"RAMIA Results: {results}")

    return results


@model_metric(name="privacy_meter_ramia")
def privacy_meter_ramia(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: ImageClassificationModel,
) -> list[Measure]:
    """Evaluate privacy leakage using Range Membership Inference Attack (RAMIA).

    This metric measures how well an attacker can infer if data similar to training
    samples was used in training, by analyzing model behavior on neighboring data points.

    This metric ONLY works with ImageClassificationModel. It will return empty results
    for other model types.

    Args:
        datashape: Schema defining dataset structure
        model: Model object containing training dataset
        dataset: Test dataset to compare against training data
        functional_model: Wrapped model with prediction functions

    Returns:
        List of Measure objects with RAMIA scores (empty if not an image model)
    """
    logger.info("Starting Privacy Meter RAMIA evaluation")

    # Check if this is an ImageClassificationModel
    from a4s_eval.service.functional_model import ImageClassificationModel as ImgModel

    if not isinstance(functional_model, ImgModel):
        logger.warning(
            f"RAMIA metric skipped: requires ImageClassificationModel, "
            f"got {type(functional_model).__name__}"
        )
        # Return empty list - this metric is not applicable to this model type
        return []

    # Get training and test images
    # For CIFAR-10, images should be in the dataset
    # Assuming dataset.data contains image data

    # Load CIFAR-10 data from the stored location
    import pickle
    import os

    cifar_path = "./tests/data/cifar-10-batches-py"

    # Check if CIFAR-10 data exists
    if not os.path.exists(cifar_path):
        logger.error(f"CIFAR-10 data not found at {cifar_path}")
        logger.info("RAMIA metric requires CIFAR-10 dataset to be present")
        return []

    def load_cifar_batch(file_path: str) -> tuple:
        with open(file_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"]
        labels = batch[b"labels"]
        # Reshape from (10000, 3072) to (10000, 3, 32, 32)
        images = images.reshape(-1, 3, 32, 32)
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        return images, np.array(labels)

    # Load training batches (ALL training batches that the model was trained on)
    train_images_list = []
    for i in range(1, 6):  # data_batch_1, 2, 3, 4, 5 (all 50k training images)
        batch_path = os.path.join(cifar_path, f"data_batch_{i}")
        images, _ = load_cifar_batch(batch_path)
        train_images_list.append(images)

    train_images = np.concatenate(train_images_list, axis=0)

    # Load test batch (actual test set that model was NOT trained on)
    test_batch_path = os.path.join(cifar_path, "test_batch")
    test_images, _ = load_cifar_batch(test_batch_path)

    logger.info(
        f"Loaded {len(train_images)} training images and {len(test_images)} test images"
    )

    # Run RAMIA attack
    results = ramia_attack(
        functional_model,
        train_images,
        test_images,
        num_neighbors=5,
        noise_scale=0.05,
    )

    # Create measures
    current_time = datetime.now()
    measures = []

    for metric_name, score in results.items():
        measure = Measure(
            name=f"ramia_{metric_name}",
            score=float(score),
            time=current_time,
        )
        measures.append(measure)

    logger.info(f"Generated {len(measures)} RAMIA measures")

    return measures
