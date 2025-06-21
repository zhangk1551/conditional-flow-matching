import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchcfm.utils import sample_8gaussians, sample_moons


class GaussiansMoonsDataset(Dataset):
    def __init__(self, batch_size=256):
        super().__init__()
        self.batch_size = batch_size

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return sample_8gaussians(self.batch_size), sample_moons(self.batch_size)


class MnistDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        e = torch.randn_like(x)
        return e, x


class MeanFlowDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_datasets(self):
        self.train_dataset = GaussiansMoonsDataset(batch_size=self.config.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=0)
#        return DataLoader(self.train_dataset, batch_size=self.config.batch_size,
#                          drop_last=True, num_workers=0)
