"""Tests for RAMIA privacy metric."""

import uuid
import pandas as pd
import pytest

from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.service.functional_model import ImageClassificationModel
from a4s_eval.service.model_factory import load_model
from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
    Model,
    ModelConfig,
    ModelFramework,
    ModelTask,
)
from tests.save_measures_utils import save_measures


@pytest.fixture
def data_shape() -> DataShape:
    """Create data shape for CIFAR-10 images."""
    metadata = pd.read_csv("tests/data/cifar10_metadata.csv").to_dict(orient="records")

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [item for item in metadata if item.get("name") == "image"],
        "target": next(rec for rec in metadata if rec.get("name") == "label"),
        "date": next(rec for rec in metadata if rec.get("name") == "date"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(data_shape: DataShape) -> Dataset:
    """Create test dataset for CIFAR-10."""
    # Create a dummy DataFrame (actual images loaded in metric)
    data = pd.DataFrame(
        {
            "image": ["placeholder"] * 100,
            "label": [0] * 100,
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(data_shape: DataShape) -> Dataset:
    """Create reference dataset for CIFAR-10."""
    data = pd.DataFrame(
        {
            "image": ["placeholder"] * 100,
            "label": [0] * 100,
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    """Create reference model."""
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture
def functional_model() -> ImageClassificationModel:
    """Load the ResNet20 CIFAR-10 model."""
    model_config = ModelConfig(
        path="./tests/data/resnet20_cifar10_jit.pt",
        framework=ModelFramework.TORCH,
        task=ModelTask.IMAGE_CLASSIFICATION,
    )

    model = load_model(model_config)
    if not isinstance(model, ImageClassificationModel):
        raise TypeError(
            f"Model type error {type(model)} != ImageClassificationModel."
        )
    return model


def test_ramia_metric_exists():
    """Test that RAMIA metric is registered."""
    functions = model_metric_registry.get_functions()
    assert "privacy_meter_ramia" in functions, "RAMIA metric not registered"


def test_ramia_metric_execution(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: ImageClassificationModel,
):
    """Test RAMIA metric execution."""
    functions = model_metric_registry.get_functions()
    ramia_func = functions["privacy_meter_ramia"]

    measures = ramia_func(data_shape, ref_model, test_dataset, functional_model)

    # Save measures for analysis
    save_measures("privacy_meter_ramia", measures)

    # Verify we got results
    assert len(measures) > 0, "No measures returned"

    # Check that we have the expected metrics
    metric_names = [m.name for m in measures]
    assert "ramia_attack_accuracy" in metric_names
    assert "ramia_privacy_leakage" in metric_names
    assert "ramia_precision" in metric_names
    assert "ramia_recall" in metric_names

    # Verify scores are in valid ranges
    for measure in measures:
        assert measure.score is not None
        assert isinstance(measure.score, (int, float))
        # Most metrics should be between 0 and 1 or reasonable integers
        if "true_" in measure.name or "false_" in measure.name:
            assert measure.score >= 0  # Count metrics
        else:
            assert -1 <= measure.score <= 1  # Probability/score metrics

    print(f"\n✓ RAMIA metric executed successfully with {len(measures)} measures")
    for measure in measures:
        print(f"  - {measure.name}: {measure.score:.4f}")
