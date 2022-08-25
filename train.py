import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.data import GOSDataModule
from model.isnet import ISNet


def train():
    # Create data loaders
    print('Loading data...')
    data_module = GOSDataModule(args.data_dir, args.batch_size, args.workers)

    # Model
    print('Creating model...')
    model = ISNet(3, 1, args)

    # Training
    print('Started training')
    accelerator = 'cpu'
    devices = 1
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = min(torch.cuda.device_count(), len(args.gpus.split(',')))
    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=args.epochs,
        callbacks=[EarlyStopping(
            monitor='val_f1_score', mode='max', patience=3, verbose=True)]
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--gpus', type=str, default='0',
                        help='A list of GPUs e.g 0,1,2,3. Use None for CPU only')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size. If training on n GPUs, it should be set'
                        ' n times larger than single GPU training.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='The number of training epochs.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--outdir', type=str, default='output',
                        help='Training output directory.')

    # Data
    parser.add_argument('--data_dir', type=str, help='Path to root data dir')
    args = parser.parse_args()
    train()
