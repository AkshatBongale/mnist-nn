"""Load the MNIST dataset and take a peek at it.

Run this file on its own to download the data and print a real digit
straight into your terminal:

    uv run data.py
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MNIST pixels are 0-255. ToTensor rescales them to 0.0-1.0, and Normalize
# shifts them to have MNIST's known mean/std so training is a little smoother.
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_dataloaders(batch_size: int = 64, data_dir: str = "./data"):
    """Return (train_loader, test_loader) for MNIST, downloading if needed.

    A DataLoader hands us the images in shuffled batches during training.
    """
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=_transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def print_digit_ascii(image_tensor, label):
    """Print one 28x28 digit to the terminal as ASCII art."""
    chars = " .:-=+*#%@"
    img = image_tensor.squeeze(0)  # (1, 28, 28) -> (28, 28)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # back to 0..1 for display
    print(f"\nThis image is labeled: {label}\n")
    for row in img:
        print("".join(chars[int(p.item() * (len(chars) - 1))] for p in row))


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"training images: {len(train_loader.dataset)}")
    print(f"test images:     {len(test_loader.dataset)}")
    print(f"batches per epoch: {len(train_loader)} (of {train_loader.batch_size} images each)")

    # Grab one batch and look at the very first image in it.
    images, labels = next(iter(train_loader))
    print(f"\none batch of images has shape: {tuple(images.shape)}  (batch, channels, height, width)")
    print_digit_ascii(images[0], labels[0].item())
