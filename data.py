import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl


class PositiveData(Dataset):
    def __init__(self, num_samples, sr=16000, seg_length=2**10, fmin=100, fmax=8000):
        self.sr = sr
        self.seg_length = seg_length
        self.freq_list = np.logspace(
            np.log2(fmin),
            np.log2(fmax),
            num_samples,
            base=2
        )

    def __len__(self):
        return len(self.freq_list)
    
    def __getitem__(self, id):
        freq = self.freq_list[id]
        x = torch.zeros(self.seg_length)
        step = int(self.sr // freq)
        x[::step * 2] = 1
        x[step::step * 2] = -1
        return x.unsqueeze(0), torch.tensor(1, dtype=torch.float32)


class NegativeData(Dataset):
    def __init__(self, num_samples, sr=16000, seg_length=2**10, fmin=100, fmax=8000):
        self.sr = sr
        self.seg_length = seg_length
        self.freq_list = np.logspace(
            np.log2(fmin),
            np.log2(fmax),
            num_samples,
            base=2
        )
        self.num_samples = num_samples

    def __len__(self):
        return len(self.freq_list)
    
    def __getitem__(self, id):
        freq = self.freq_list[id]
        x = torch.zeros(self.seg_length)
        step = int(self.sr // freq)
        x[::step * 4] = 1
        x[step::step * 4] = 1
        x[2 * step::step * 4] = -1
        x[3 * step::step * 4] = -1
        return x.unsqueeze(0), torch.tensor(0, dtype=torch.float32)

    

class DataMoudle(pl.LightningDataModule):
    def __init__(self, num_samples, batch_size, sr=16000, seg_length=2**10, fmin=100, fmax=8000, **kwargs):
        super().__init__()
        positive_data = PositiveData(num_samples // 2, sr, seg_length, fmin, fmax)
        negative_data = NegativeData(num_samples - num_samples // 2, sr, seg_length, fmin, fmax)
        self.dataset = torch.utils.data.ConcatDataset([positive_data, negative_data])
        self.batch_size = batch_size
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset")
        parser.add_argument('--num_samples', type=int, default=1000)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--sr', type=int, default=16000)
        parser.add_argument('--seg_length', type=int, default=2**10)
        parser.add_argument('--fmin', type=int, default=140)
        parser.add_argument('--fmax', type=int, default=8000)
        return parent_parser

    def setup(self, stage=None):
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)