import os
from pathlib import Path
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
import pytorch_lightning as pl

import numpy as np
from skimage import io


INPUT_SIZE = [1024, 1024]


class GOSRandomHFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSResize:
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # import time
        # start = time.time()

        image = torch.squeeze(F.upsample(torch.unsqueeze(
            image, 0), self.size, mode='bilinear'), dim=0)
        label = torch.squeeze(F.upsample(torch.unsqueeze(
            label, 0), self.size, mode='bilinear'), dim=0)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSRandomCrop:
    def __init__(self, size=[288, 288]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top:top+new_h, left:left+new_w]
        label = label[:, top:top+new_h, left:left+new_w]

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSNormalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image, self.mean, self.std)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


def preprocess_image(im):
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)

    im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
    im_tensor = torch.transpose(torch.transpose(im_tensor, 1, 2), 0, 1)
    # if len(size) < 2:
    #     return im_tensor, im.shape[0:2]
    # else:
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = F.upsample(im_tensor, INPUT_SIZE, mode="bilinear")
    im_tensor = torch.squeeze(im_tensor, 0)

    return im_tensor.type(torch.uint8), im.shape[0:2]


def preprocess_gt(gt):
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.uint8), 0)

    # if (len(size) < 2):
    #     return gt_tensor.type(torch.uint8), gt.shape[0:2]
    # else:
    gt_tensor = torch.unsqueeze(gt_tensor.type(torch.float32), 0)
    gt_tensor = F.upsample(gt_tensor, INPUT_SIZE, mode="bilinear")
    gt_tensor = torch.squeeze(gt_tensor, 0)

    return gt_tensor.type(torch.uint8), gt.shape[0:2]


def collate_fn(batch):
    batch_dict = OrderedDict()
    for item in batch:
        for k, v in item.items():
            if k not in batch_dict:
                batch_dict[k] = []
            batch_dict[k].append(v)

    for k, v in batch_dict.items():
        batch_dict[k] = torch.stack(v)
    return batch_dict


def list_images(input_dir):
    return sorted([os.path.join(input_dir, name)
                   for name in os.listdir(input_dir)])


class GOSDataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.transform = transform

        image_dir = os.path.join(data_dir, 'images')
        gt_dir = os.path.join(data_dir, 'masks')
        self.image_paths = list_images(image_dir)[:300]
        self.gt_paths = list_images(gt_dir)[:300]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        image = io.imread(image_path)
        gt = io.imread(gt_path)

        # Preprocess
        image, image_shape = preprocess_image(image)
        gt, _ = preprocess_gt(gt)

        # Transform
        image = torch.divide(image, 255.)
        gt = torch.divide(gt, 255.)
        imidx = torch.from_numpy(np.array(idx))
        image_shape = torch.from_numpy(np.array(image_shape))
        sample = {'imidx': imidx, 'image': image,
                  'label': gt, 'shape': image_shape}
        if self.transform:
            sample = self.transform(sample)
        return sample


class GOSDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 8, workers: int = 8):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        self.batch_size = batch_size
        self.workers = workers
        self.train_transform = transforms.Compose([
            GOSRandomHFlip(),
            GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ])
        self.val_transform = transforms.Compose([
            GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ])

    def train_dataloader(self):
        dataset = GOSDataset(self.train_dir, self.train_transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers)

    def val_dataloader(self):
        dataset = GOSDataset(self.val_dir, self.val_transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers)
