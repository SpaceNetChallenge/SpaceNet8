from omegaconf import DictConfig
from torch.utils.data import DataLoader

# isort: off
from spacenet8_model.datasets.spacenet8 import SpaceNet8Dataset, SpaceNet8TestDataset
from spacenet8_model.datasets.transforms import get_transforms, get_test_transforms
# isort: on


def get_dataloader(config: DictConfig, is_train: bool) -> DataLoader:
    transforms = get_transforms(config, is_train)

    if is_train:
        batch_size = config.Dataloader.train_batch_size
        num_workers = config.Dataloader.train_num_workers
        shuffle = True
    else:
        batch_size = config.Dataloader.val_batch_size
        num_workers = config.Dataloader.val_num_workers
        shuffle = False

    dataset = SpaceNet8Dataset(config, is_train, transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)


def get_test_dataloader(config: DictConfig, test_to_val=False, tta_hflip=False, tta_vflip=False) -> DataLoader:
    transforms = get_test_transforms(config, tta_hflip=tta_hflip, tta_vflip=tta_vflip)

    dataset = SpaceNet8TestDataset(config, transforms, test_to_val)

    return DataLoader(
        dataset,
        batch_size=config.Dataloader.test_batch_size,
        shuffle=False,
        num_workers=config.Dataloader.test_num_workers)
