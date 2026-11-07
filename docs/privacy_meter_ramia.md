## Privacy Meter : RAMIA (Range Membership Inference Attack)

## A description of the metric:

Library: Privacy Meter (https://github.com/privacytrustlab/ml_privacy_meter)

RAMIA tests whether a model leaks information about its training data. The core idea is that the models tend to be more "confident" on data they were trained on. RAMIA goes a step further and checks if the model is also confident on similar data (neighbors- a “range”) to training samples. Unlike traditional Membership Inference Attacks (MIA) which check for exact membership, RAMIA analyzes the model's behavior in the neighborhood of training data points.

## How it works:
1.Take a sample image.
2.Generate slightly perturbed versions (add small noise).
3.Compare model confidence on original vs. perturbed images.
4.Training samples typically show smaller confidence drops on neighbors (the model "recognizes" the neighborhood).
5.Non-training samples show larger confidence drops (unfamiliar territory).


This matters because if an attacker can determine that certain data (or similar data) was used for training, they can potentially:
- Identify individuals in the training set.
- Infer sensitive information about training data.
- Violate data privacy regulations (GDPR, etc.).

Example: A medical model trained on patient X-rays - RAMIA could reveal if a specific patient's data was used for training, which is a privacy violation.

- Model Type -> Image Classification (PyTorch) - ResNet20
- Model Format ->	TorchScript JIT (.pt)
- Data Type ->	Images (CIFAR-10 format: 32x32 RGB)  Download: https://www.cs.toronto.edu/~kriz/cifar.html 
- Input Shape ->	(N, 3, 32, 32) - channels first
- Input Range ->	Normalized to [0, 1]
- Model Type ->	Image Classification (PyTorch)

## High Level Implementation Overview of RAMIA

To implement RAMIA we had to do the following:

1. Core Metric Function
   - The actual RAMIA algorithm implementation
2. Image Model Support
   - A4S only supported tabular data, so we need image support
3. Model Loading
   - Ability to load PyTorch image models
4. Test Suite
   - Tests to verify everything works
5. Data & Models
   - Test data (CIFAR-10 images)
   - Test model (ResNet20)
6. Documentation & Visualization
   - Explain your work
   - Show results in graphs