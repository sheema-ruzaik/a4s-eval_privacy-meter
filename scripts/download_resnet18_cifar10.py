"""
Script to download a pre-trained ResNet18 model for CIFAR-10
This model is pre-trained on CIFAR-10 dataset from PyTorch Hub
"""

import torch
import torchvision.models as models

def download_resnet18_cifar10():
    """Download pre-trained ResNet18 model for CIFAR-10 from PyTorch Hub"""

    print("Downloading pre-trained ResNet18 model for CIFAR-10...")

    try:
        # Download from PyTorch Hub (chenyaofo maintains pre-trained CIFAR models)
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            pretrained=True
        )

        print("✓ Model downloaded successfully!")
        print(f"Model type: {type(model)}")

        # Save the model
        output_path = "./tests/data/resnet20_cifar10.pt"
        torch.save(model, output_path)

        print(f"✓ Model saved to: {output_path}")

        # Also save as TorchScript for compatibility with your existing infrastructure
        model.eval()
        example_input = torch.randn(1, 3, 32, 32)
        traced_model = torch.jit.trace(model, example_input)

        output_path_jit = "./tests/data/resnet20_cifar10_jit.pt"
        torch.jit.save(traced_model, output_path_jit)

        print(f"✓ TorchScript model saved to: {output_path_jit}")

        # Test the model
        print("\nTesting model...")
        with torch.no_grad():
            output = model(example_input)
            print(f"Output shape: {output.shape}")
            print(f"Number of classes: {output.shape[1]}")
            print(f"Prediction: Class {torch.argmax(output, dim=1).item()}")

        print("\n✓ All done! ResNet20 model ready for CIFAR-10.")

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\nFalling back to torchvision ResNet18...")

        # Fallback: Use standard ResNet18 and modify for CIFAR-10
        model = models.resnet18(pretrained=False)

        # Modify for CIFAR-10 (32x32 images, 10 classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)

        # Save the model
        output_path = "./tests/data/resnet18_cifar10.pt"
        torch.save(model, output_path)
        print(f"✓ Custom ResNet18 saved to: {output_path}")
        print("⚠ Note: This model is NOT pre-trained. You may need to train it.")

if __name__ == "__main__":
    download_resnet18_cifar10()
