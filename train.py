import argparse

import pytorch_lightning as pl

from isnet.dataloader import create_dataloader
from isnet.model import ISNetModule


def main():
    # data
    train_loader = create_dataloader(args.data_dir)
    val_loader = create_dataloader(args.data_dir)

    # model
    model = ISNetModule()

    # training
    gpus = len(args.device.split(','))
    trainer = pl.Trainer(gpus=gpus)
    trainer.fit(model, train_loader, val_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--nosave', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main()
