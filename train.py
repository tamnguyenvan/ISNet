import os
import argparse

import pytorch_lightning as pl

from utils.data import GOSDataModule
from model.isnet import ISNet


def train():
    # Create data loaders
    print('Loading data...')
    data_module = GOSDataModule(args.data_dir, args.batch_size, args.workers)

    # Model
    print('Creating model...')
    model = ISNet()

    # Training
    print('Started training')
    trainer = pl.Trainer(
        max_epochs=1000,
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size. If training on n GPUs, it should be set'
                        ' n times larger than single GPU training.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers.')

    # Data
    parser.add_argument('--data_dir', type=str, help='Path to root data dir')
    args = parser.parse_args()
    train()
