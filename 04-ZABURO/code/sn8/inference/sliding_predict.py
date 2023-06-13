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


def sliding_predict(
    model: nn.Module,
    out_dir: Path,
    dataset: torch.utils.data.Dataset,
    image_paths: Sequence[Path],
    crop_size: int,
    d4_tta: bool,
    batch_size: int,
    device: DeviceLike,
    num_workers: int = 0,
    pin_memory: bool = False,
    progress: bool = True,
) -> None:
    assert not hasattr(dataset, "pad") or dataset.pad is None

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

            img_shape = next(iter(batch.values())).shape[-2:]
            assert all(img.shape[-2:] == img_shape for img in batch.values())
            assert crop_size <= img_shape[0] and crop_size <= img_shape[1]

            y_pred_total = None
            counts = torch.zeros(img_shape)

            nrows = (img_shape[0] + crop_size - 1) // crop_size
            ncols = (img_shape[1] + crop_size - 1) // crop_size

            ys, xs = np.meshgrid(
                np.linspace(0, img_shape[0] - crop_size, nrows).astype(int),
                np.linspace(0, img_shape[1] - crop_size, ncols).astype(int),
            )
            for y, x in zip(ys.ravel(), xs.ravel()):
                crop_batch = {key: img[..., y : y + crop_size, x : x + crop_size] for key, img in batch.items()}
                for transform, transform_inv in zip(D4, D4_inv) if d4_tta else zip(D4[:1], D4_inv[:1]):
                    y_pred = model(**{key: transform(img) for key, img in crop_batch.items()})

                    assert isinstance(y_pred, Tensor)
                    assert y_pred.dim() == 4

                    y_pred = transform_inv(y_pred)
                    y_pred = y_pred.sigmoid().detach().cpu()

                    if y_pred_total is None:
                        y_pred_total = torch.zeros(y_pred.shape[:-2] + img_shape)

                    y_pred_total[..., y : y + crop_size, x : x + crop_size] += y_pred
                    counts[..., y : y + crop_size, x : x + crop_size] += 1

            assert y_pred_total is not None
            assert (counts >= 1).all()
            y_pred_total /= counts
            y_pred = y_pred_total

            for i in range(y_pred.shape[0]):
                mask = y_pred[i].numpy()
                mask = (mask * 255).astype(np.uint8)  # (C, H, W)

                image_ds = gdal.Open(str(image_paths[index]))

                mask_ds = gdal.GetDriverByName("GTiff").Create(
                    str(out_dir / image_paths[index].name),
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
