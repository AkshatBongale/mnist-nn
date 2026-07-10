"""Train the network on MNIST and watch it learn.

    uv run train.py

You should see the test accuracy jump from ~10% (random) to ~97% in a
couple of epochs.
"""

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloaders
from model import MLP


def pick_device():
    """Use the fastest backend available.

    - CUDA  -> NVIDIA GPU (common on Linux/Windows)
    - MPS   -> Apple Silicon GPU (M1/M2/M3 Macs)
    - CPU   -> always works; totally fine for this small model
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()  # no gradients needed when we're just checking accuracy
def accuracy(model, loader, device):
    """Fraction of images the model classifies correctly."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(epochs: int = 3, lr: float = 1e-3):
    device = pick_device()
    print(f"training on: {device}\n")

    train_loader, test_loader = get_dataloaders(batch_size=64)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()                       # measures "how wrong"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # does the "nudge"

    # TensorBoard writes numbers to a "runs/" folder as we train, so we can
    # watch the loss curve live in a browser. Each run lands in its own
    # timestamped subfolder, so you can compare different settings side by side.
    writer = SummaryWriter()
    step = 0  # counts every batch we've trained on, for the x-axis of the graphs

    start_acc = accuracy(model, test_loader, device)
    print(f"accuracy BEFORE training: {start_acc:.1%}  (~random guessing)\n")
    writer.add_scalar("accuracy/test", start_acc, step)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            scores = model(images)          # 1. forward:  predict
            loss = loss_fn(scores, labels)  # 2. loss:     how wrong

            optimizer.zero_grad()           #    clear last step's gradients
            loss.backward()                 # 3. backward: gradients (backprop)
            optimizer.step()                # 4. step:     nudge the weights

            step += 1
            writer.add_scalar("loss/train", loss.item(), step)  # -> TensorBoard

            if batch % 200 == 0:
                print(f"epoch {epoch}  batch {batch:4d}/{len(train_loader)}  loss {loss.item():.3f}")

        acc = accuracy(model, test_loader, device)
        writer.add_scalar("accuracy/test", acc, step)
        print(f">>> epoch {epoch} done  |  test accuracy: {acc:.1%}\n")

    writer.close()
    torch.save(model.state_dict(), "mnist_model.pt")
    print("saved the trained model to mnist_model.pt")
    print("\nSee the curves:  uv run tensorboard --logdir runs   (then open http://localhost:6006)")


if __name__ == "__main__":
    train()
