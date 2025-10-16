import pytorch_lightning as pl 
import torch 
from torch import nn 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

# Lightning Module 
class LitNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)  # Lightning handles logging
        return loss 
    
    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
# dataset and trainer 
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.MNIST(root="data", train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

trainer = pl.Trainer(max_epoch=5)
trainer.fit(LitNet(), train_loader)
    