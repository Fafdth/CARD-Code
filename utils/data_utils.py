"""Dataset splits and paired loaders for the five datasets in paper Table I."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS

from .my_dataset import MyDataSet, CarsDataset, StanfordDogs, NABirdsDataset


def load_dataset(root, train_folder="train", val_folder="val"):
    """Read pre-split image folders with a shared class-to-index mapping."""
    root = Path(root)
    train_path, val_path = root / train_folder, root / val_folder
    if not val_path.is_dir() and val_folder == "val":
        val_path = root / "test"
    classes = sorted(p.name for p in train_path.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No training classes under {train_path}")
    mapping = {name: index for index, name in enumerate(classes)}

    def samples(path):
        images, labels = [], []
        for folder in sorted(path.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name not in mapping:
                raise ValueError(f"Unknown test class {folder.name!r}; training/test labels must agree.")
            paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
            images.extend(str(p) for p in paths)
            labels.extend([mapping[folder.name]] * len(paths))
        if not images:
            raise ValueError(f"No images under {path}")
        return images, labels

    train_images, train_labels = samples(train_path)
    test_images, test_labels = samples(val_path)
    return train_images, train_labels, test_images, test_labels


def _cub_splits(root):
    """Honor CUB's official image IDs, class labels and train/test split."""
    root = Path(root)
    def rows(filename):
        return [line.strip().split(maxsplit=1) for line in (root / filename).read_text().splitlines() if line.strip()]
    paths = {int(key): str(root / "images" / value) for key, value in rows("images.txt")}
    labels = {int(key): int(value) - 1 for key, value in rows("image_class_labels.txt")}
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for key, is_train in rows("train_test_split.txt"):
        key = int(key)
        images, targets = (train_images, train_labels) if int(is_train) else (test_images, test_labels)
        images.append(paths[key])
        targets.append(labels[key])
    return train_images, train_labels, test_images, test_labels


def get_loader(args):
    root = Path(args.data_root)
    if args.dataset in {"CUB_200_2011", "RP_281"}:
        if args.dataset == "CUB_200_2011" and (root / "train_test_split.txt").is_file():
            train_images, train_labels, test_images, test_labels = _cub_splits(root)
        else:
            train_images, train_labels, test_images, test_labels = load_dataset(root)
        trainset = MyDataSet(train_images, train_labels, args)
        testset = MyDataSet(test_images, test_labels, args, is_test=True)
    elif args.dataset == "car":
        trainset = CarsDataset(str(root / "devkit/cars_train_annos.mat"), str(root / "cars_train"),
                               args)
        testset = CarsDataset(str(root / "cars_test_annos_withlabels.mat"), str(root / "cars_test"),
                              args, is_test=True)
    elif args.dataset == "dog":
        trainset = StanfordDogs(str(root), args)
        testset = StanfordDogs(str(root), args, is_test=True)
    elif args.dataset == "nabirds":
        trainset = NABirdsDataset(str(root), args)
        testset = NABirdsDataset(str(root), args, is_test=True)
    else:
        raise ValueError("Unknown paper dataset: " + args.dataset)

    micro_batch = getattr(args, "micro_batch_size", None) or args.train_batch_size
    workers = getattr(args, "num_workers", 4)
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(trainset, batch_size=micro_batch, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, **common)
    if len(train_loader) == 0 or len(testset) == 0:
        raise ValueError("Empty loader: reduce the micro-batch size or check dataset splits.")
    return train_loader, test_loader


def calibration_loader(train_loader):
    """Visit every training image once, including the final partial batch."""
    return DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                      shuffle=False, drop_last=False, num_workers=train_loader.num_workers,
                      pin_memory=train_loader.pin_memory)
