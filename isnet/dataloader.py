import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.image_dir = Path(root_dir) / 'images'
        self.label_dir = Path(root_dir) / 'labels'
        self.label_paths = self._load_paths(self.label_dir)

    def _load_paths(self, data_dir: str):
        paths = [str(p) for p in Path(data_dir).glob(f'*.jpg')]
        return paths

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        image_path = label_path.replace('/labels', '/images')
        img = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        return img, label


def create_dataloader(
    root_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    workers: int = 4
):
    dataset = CustomDataset(root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers
    )
