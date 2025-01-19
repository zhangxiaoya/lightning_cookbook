import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger as pl_logger
import torch.nn as nn
import lightning as L
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        return self.layers(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.example_input_array=torch.Tensor(32, 1, 28 * 28)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.encoder(x)

root_dir = '/Users/ryancheung/workspace/data'
data_dir = os.path.join(root_dir, 'data')
dataset = MNIST(data_dir, download=True, train=True,transform=ToTensor())
data_loader = DataLoader(dataset, batch_size=32)

logger_dir = os.path.join(root_dir, 'logger')
logger = pl_logger(logger_dir)
trainer = L.Trainer(limit_train_batches=10, logger=logger, max_epochs=5, profiler='advanced')

model = LitAutoEncoder(Encoder(), Decoder())
trainer.fit(model=model, train_dataloaders=data_loader)
print(model)