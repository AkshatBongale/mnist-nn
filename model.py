import torch 
from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    model = MLP()
    print(model)
    fake_image = torch.randn(1,1,28,28)
    output = model(fake_image)
    print(f"\noutput scores shape: {tuple(output.shape)}  (one score per digit)")
    print(f"predicted digit: {output.argmax(dim=1).item()}  <- random weights, so it's just a guess")