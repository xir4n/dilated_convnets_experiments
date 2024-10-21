import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from data import DataMoudle
import models


HYPERPARAMS = { #for test only
    'dataset': {
        'num_samples': 100,
        'batch_size': 25,
        'sr': 16000,
        'seg_length': 2**10,
        'fmin': 140,
        'fmax': 8000,
    },
    'model': {
        'Q': 6,
        'T': 4,
        'J': 6,
        'lr': 1e-1,
    },
}

def train(args):
    dict_args = vars(args)
    # Create a folder to save the model
    save_path = os.path.join(args.save_foler, args.arch)
    os.makedirs(save_path, exist_ok=True)

    # Create the dataset
    dataset = DataMoudle(**dict_args)

    # Construct the model
    constructor = getattr(models, args.arch)
    pl_module = constructor(**dict_args)

    # get example input
    dataset.setup() 
    pl_module.example_input_array = next(iter(dataset.train_dataloader()))[0]

    # Setup checkpoints and Tensorboard logger
    checkpoint_cb = ModelCheckpoint(
        dirpath=save_path,
        monitor="val_loss",
        save_last=True,
        filename="best",
        save_weights_only=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=save_path,
        name='logs',
        log_graph=args.log_graph,
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=tb_logger, 
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        callbacks=[checkpoint_cb])

    # Train the model
    trainer.fit(pl_module, dataset)
    trainer.test(pl_module, dataset, verbose=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_foler', type=str, default='/Users/zhang/MuReNN/data/dilated_conv1d')
    parser.add_argument('--arch', type=str, default='MuReNN')
    # dataset hyperparameters
    parser = DataMoudle.add_data_specific_args(parser)
    # model hyperparameters
    parser = models.Plmodel.add_model_specific_args(parser)
    # logger hyperparameters
    parser.add_argument('--log_graph', type=bool, default=False)
    # trainer hyperparameters
    parser.add_argument('--max_epochs', type=int, default=300)
    args = parser.parse_args()
    train(args)

