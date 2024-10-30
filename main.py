import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from data import DataMoudle
import models


HYPERPARAMS = { #for test only
    'dataset': {
        'num_samples': 1000,
        'batch_size': 25,
        'seg_length': 2**10,
        'step_min': 1,
        'step_max': 16,
    },
    'model': {
        'Q': 6,
        'T': 3,
        'J': 6,
        'lr': 1e-1,
        'scale_factor': 1,
    },
}

def train(args):
    dict_args = vars(args)
    # Create a folder to save the model
    os.makedirs(args.save_foler, exist_ok=True)

    # Create the dataset
    dataset = DataMoudle(**dict_args)

    # Construct the model
    constructor = getattr(models, args.arch)
    pl_module = constructor(**dict_args)

    # Setup checkpoints and Tensorboard logger
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_foler,
        monitor="val_loss",
        save_last=True,
        filename="best",
        save_weights_only=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save_foler,
        name=f'{args.arch}_logs',
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=tb_logger, 
        callbacks=[checkpoint_cb],
    )

    # Train the model
    if not args.test:
        trainer.fit(pl_module, dataset)
        trainer.test(pl_module, dataset, verbose=False)
    else:
        trainer.test(pl_module, dataset, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_foler', type=str, default='/Users/zhang/MuReNN/data/dilated_conv1d')
    parser.add_argument('--arch', type=str, default='MuReNN')
    parser.add_argument('--test', type=bool, default=False, help='if True, run test only')
    parser.add_argument('--ckpt_path', type=str, default=None)
    # dataset hyperparameters
    parser = DataMoudle.add_data_specific_args(parser)
    # model hyperparameters
    parser = models.Plmodel.add_model_specific_args(parser)
    # trainer hyperparameters
    parser.add_argument('--max_epochs', type=int, default=600)
    args = parser.parse_args()
    train(args)

