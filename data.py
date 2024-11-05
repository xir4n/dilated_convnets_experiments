import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl


class DataMoudle(pl.LightningDataModule):
    def __init__(self, num_samples, batch_size, seg_length=2**10, step_min=1, step_max=2**6, **kwargs):
        super().__init__()
        self.positive_data = PositiveData(num_samples // 2, seg_length, step_min, step_max)
        self.negative_data = NegativeData(num_samples - num_samples // 2, seg_length, step_min, step_max)
        self.dataset = torch.utils.data.ConcatDataset([self.positive_data, self.negative_data])
        self.batch_size = batch_size
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset")
        parser.add_argument('--num_samples', type=int, default=1000)
        parser.add_argument('--batch_size', type=int, default=256)
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


class PositiveData(DataMoudle):
    def __init__(self, num_samples, seg_length, step_min, step_max):
        self.seg_length = seg_length
        self.step_list = np.logspace(np.log2(step_min), np.log2(step_max), num_samples, dtype=int, base=2)
        # for i, step in enumerate(self.step_list):
        #     if step % 2 != 0:
        #         self.step_list[i] += 1

    def __len__(self):
        return len(self.step_list)
    
    def __getitem__(self, id):
        step = self.step_list[id]
        x = torch.zeros(self.seg_length)
        x[::step * 2] = 1
        x[step::step * 2] = -1
        tau = np.random.randint(0, self.seg_length)
        x = x.roll(tau)
        # n = torch.randn(self.seg_length) / (step)
        # x += n
        return x.unsqueeze(0), torch.tensor(1, dtype=torch.float32)


class NegativeData(DataMoudle):
    def __init__(self, num_samples, seg_length, step_min, step_max):
        self.seg_length = seg_length
        self.step_list = np.logspace(np.log2(step_min), np.log2(step_max), num_samples, dtype=int, base=2)
        # for i, step in enumerate(self.step_list):
        #     if step % 2 != 0:
        #         self.step_list[i] += 1

    def __len__(self):
        return len(self.step_list)
    
    def __getitem__(self, id):
        step = self.step_list[id]
        x = torch.zeros(self.seg_length)
        x[::step * 4] = 1
        x[step::step * 4] = 1
        x[2 * step::step * 4] = -1
        x[3 * step::step * 4] = -1
        tau = np.random.randint(0, self.seg_length)
        x = x.roll(tau)
        # n = torch.randn(self.seg_length) / (step)
        # x += n
        return x.unsqueeze(0), torch.tensor(0, dtype=torch.float32)
