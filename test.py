from models import *
from data import DataMoudle
from main import HYPERPARAMS

dataset = DataMoudle(**HYPERPARAMS['dataset'])
x, y = dataset.dataset[0]
x = x.unsqueeze(0)
murenn = MuReNN(**HYPERPARAMS['model'])
murenn(x)

conv1d = Conv1D(**HYPERPARAMS['model'])
conv1d(x)

wavenet = WaveNet(**HYPERPARAMS['model'])
wavenet(x)