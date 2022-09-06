from pathlib import Path
import argparse
import warnings

from model.isnet import ISNet
from utils.data import GOSDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
        default_root_dir=args.root_dir,
        callbacks=[
            EarlyStopping(monitor='val_f1_score', mode='max',
                          patience=20, verbose=True)
        ]
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.ckpt_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size. If training on n GPUs, it should be set'
                        ' n times larger than single GPU training.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='The number of training epochs.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--outdir', type=str, default='output',
                        help='Training output directory.')
    parser.add_argument('--root_dir', type=str, default='.',
                        help='Logs directory.')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Checkpoints directory')

    # Data
    parser.add_argument('--data_dir', type=str, help='Path to root data dir')
    args = parser.parse_args()
    train()
