from model.isnet import ISNet
from utils.data import GOSDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

import argparse
import warnings

warnings.simplefilter('ignore')


def train():
    # Create data loaders
    print('Loading data...')
    data_module = GOSDataModule(args.data_dir, args.batch_size, args.workers)

    # Model
    print('Creating model...')
    model = ISNet(3, 1, args)

    # Training
    print('Started training')
    trainer = pl.Trainer(
        devices='auto',
        accelerator='auto',
        max_epochs=args.epochs,
        callbacks=[EarlyStopping(
            monitor='val_f1_score', mode='max', patience=5, verbose=True)]
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training
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
