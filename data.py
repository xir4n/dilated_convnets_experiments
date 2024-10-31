import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl


class DataMoudle(pl.LightningDataModule):
    def __init__(self, batch_size, num_samples, seg_length, step_min, step_max, **kwargs):
        super().__init__()
        self.positive_data = PositiveData(seg_length, self.get_step_list(num_samples//2, step_min, step_max))
        self.negative_data = NegativeData(seg_length, self.get_step_list(num_samples - num_samples//2, step_min, step_max))
        self.dataset = torch.utils.data.ConcatDataset([self.positive_data, self.negative_data])
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.seg_length = seg_length
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset")
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_samples', type=int, default=1000)
        parser.add_argument('--seg_length', type=int, default=2**10)
        parser.add_argument('--step_min', type=int, default=1)
        parser.add_argument('--step_max', type=int, default=2**6)
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

    def get_step_list(self, num_samples, step_min, step_max):
        step_list = torch.linspace(step_min, step_max+1, num_samples, dtype=int)
        return step_list


class PositiveData(Dataset):
    def __init__(self, seg_length, step_list):
        self.seg_length = seg_length
        self.step_list = step_list

    def __len__(self):
        return len(self.step_list)
    
    def __getitem__(self, id):
        step = self.step_list[id]
        spacing = step % 2
        T = step * 2
        padding = self.seg_length - self.seg_length // T * T
        x = torch.zeros(self.seg_length)
        x[::step * 2] = 1
        x[step::step * 2] = -1
        x[-padding:] = 0
        n = torch.randn(self.seg_length) / (step)
        x += n
        return x.unsqueeze(0), torch.tensor(1, dtype=torch.float32), torch.tensor(spacing)


class NegativeData(Dataset):
    def __init__(self, seg_length, step_list):
        self.seg_length = seg_length
        self.step_list = step_list

    def __len__(self):
        return len(self.step_list)
    
    def __getitem__(self, id):
        step = self.step_list[id]
        spacing = step % 2
        T = step * 4
        padding = self.seg_length - self.seg_length // T * T
        x = torch.zeros(self.seg_length)
        x[::step * 4] = 1
        x[step::step * 4] = 1
        x[2 * step::step * 4] = -1
        x[3 * step::step * 4] = -1
        x[-padding:] = 0
        n = torch.randn(self.seg_length) / (step)
        x += n
        return x.unsqueeze(0), torch.tensor(0, dtype=torch.float32), torch.tensor(spacing)
