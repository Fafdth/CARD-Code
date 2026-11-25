"""Metadata readers used by the five CARD datasets (paper Table I)."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from .paired_views import PairedViewTransform


class PairedImageDataset(Dataset):
    """Apply one shared HR/LR augmentation pipeline to an indexed image list."""
    def __init__(self, samples, args, is_test=False):
        self.samples = samples
        self.views = PairedViewTransform(args, is_test=is_test)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            hr, lr = self.views(image)
        return hr, lr, int(target)


class MyDataSet(PairedImageDataset):
    """CUB / RP-281 image lists prepared by data_utils."""
    def __init__(self, images_path, images_class, args, is_test=False):
        super().__init__(list(zip(images_path, images_class)), args, is_test)


class CarsDataset(PairedImageDataset):
    def __init__(self, mat_anno, data_dir, args, is_test=False):
        annotations = loadmat(mat_anno)["annotations"].ravel()
        samples = [(Path(data_dir) / str(row["fname"].item()),
                    int(row["class"].item()) - 1) for row in annotations]
        super().__init__(samples, args, is_test)


class StanfordDogs(PairedImageDataset):
    def __init__(self, data_dir, args, is_test=False):
        root = Path(data_dir)
        image_root = root / ("Images" if (root / "Images").is_dir() else "images")
        list_root = root / "lists" if (root / "lists").is_dir() else root
        metadata = loadmat(list_root / ("test_list.mat" if is_test else "train_list.mat"))
        names = metadata.get("annotation_list", metadata.get("file_list"))
        if names is None:
            raise ValueError("Stanford Dogs split needs annotation_list or file_list.")
        samples = []
        for name, target in zip(names, metadata["labels"].ravel()):
            while isinstance(name, np.ndarray):
                name = name.item()
            name = str(name)
            if not name.lower().endswith(".jpg"):
                name += ".jpg"
            samples.append((image_root / name, int(target) - 1))
        super().__init__(samples, args, is_test)


class NABirdsDataset(PairedImageDataset):
    def __init__(self, data_dir, args, is_test=False):
        root = Path(data_dir)
        def table(filename, columns):
            return pd.read_csv(root / filename, sep=r"\s+", names=columns)
        paths = table("images.txt", ["id", "path"])
        labels = table("image_class_labels.txt", ["id", "target"])
        split = table("train_test_split.txt", ["id", "train"])
        mapping = {label: index for index, label in enumerate(sorted(labels.target.unique()))}
        rows = paths.merge(labels, on="id").merge(split, on="id")
        rows = rows[rows.train == (0 if is_test else 1)]
        samples = [(root / "images" / row.path, mapping[row.target]) for row in rows.itertuples()]
        super().__init__(samples, args, is_test)
