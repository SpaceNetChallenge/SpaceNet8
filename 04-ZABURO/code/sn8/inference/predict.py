from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from osgeo import gdal, osr
from pytorch_pfn_extras.runtime import runtime_registry
from somen.pytorch_utility.d4_transforms import D4, D4_inv
from somen.types import DeviceLike
from torch import Tensor, nn
from tqdm import tqdm


def predict(
    model: nn.Module,
    out_dir: Path,
    dataset: torch.utils.data.Dataset,
    image_paths: Sequence[Path],
    d4_tta: bool,
    batch_size: int,
    device: DeviceLike,
    num_workers: int = 0,
    pin_memory: bool = False,
    progress: bool = True,
) -> None:
    model.to(device)
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=pin_memory,
    )

    if progress:
        data_loader = tqdm(data_loader)

    runtime = runtime_registry.get_runtime_class_for_device_spec(device)(device, {})

    with torch.no_grad():

        index = 0
        for batch in data_loader:
            batch = runtime.convert_batch(batch)
            assert isinstance(batch, dict)
            assert all(img.dim() == 4 for img in batch.values())

            y_pred_total = None

            for transform, transform_inv in zip(D4, D4_inv) if d4_tta else zip(D4[:1], D4_inv[:1]):
                y_pred = model(**{key: transform(img) for key, img in batch.items()})

                if not isinstance(y_pred, dict):
                    y_pred = {"": y_pred}

                assert all(isinstance(value, Tensor) and value.dim() == 4 for value in y_pred.values())

                y_pred = {key: transform_inv(value).sigmoid().detach().cpu() for key, value in y_pred.items()}

                if y_pred_total is None:
                    y_pred_total = y_pred
                else:
                    assert set(y_pred.keys()) == set(y_pred_total.keys())
                    for key, value in y_pred.items():
                        y_pred_total[key] += value

            assert y_pred_total is not None
            y_pred = {key: value / (len(D4) if d4_tta else 1) for key, value in y_pred_total.items()}

            if dataset.pad is not None:
                pad = dataset.pad
                y_pred = {key: value[..., pad:-pad, pad:-pad] for key, value in y_pred.items()}

            for i in range(next(iter(y_pred.values())).shape[0]):
                for key, value in y_pred.items():
                    mask = value[i].detach().cpu().numpy()
                    mask = (mask * 255).clip(0, 255).astype(np.uint8)  # (C, H, W)

                    image_ds = gdal.Open(str(image_paths[index]))
                    prefix = "" if key == "" else f"{key}_"

                    mask_ds = gdal.GetDriverByName("GTiff").Create(
                        str(out_dir / (prefix + image_paths[index].name)),
                        mask.shape[2],
                        mask.shape[1],
                        mask.shape[0],
                        gdal.GDT_Byte,
                    )
                    mask_ds.SetGeoTransform(image_ds.GetGeoTransform())

                    sr = osr.SpatialReference()
                    sr.ImportFromWkt(image_ds.GetProjectionRef())
                    mask_ds.SetProjection(sr.ExportToWkt())

                    for j in range(mask.shape[0]):
                        band = mask_ds.GetRasterBand(j + 1)
                        band.WriteArray(mask[j])
                        band.FlushCache()

                    del mask_ds, image_ds
                index += 1
    assert len(dataset) == index
