import torch
from torch import nn, optim 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 


# dataset 
transform = transforms.ToTensor()
train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


# model 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLu(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)


model = Net()


# training setup 
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop (boilerplate)
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}: loss = {total_loss / len(train_loader):.4f}")
