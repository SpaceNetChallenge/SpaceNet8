import warnings
from pathlib import Path
from typing import Optional

import fire
import pandas as pd
from somen.types import PathLike

from sn8.data_prep.speed_masks import get_parameters, speed_mask


def main(
    data_dir: PathLike,
    output_conversion_csv: PathLike,
    output_mask_dir: PathLike,
    output_mask_multidim_dir: Optional[PathLike] = None,
    buffer_distance_meters: float = 2.0,
) -> None:

    data_dir = Path(data_dir)
    geojson_dir = data_dir / "annotations" / "prepped_cleaned"
    image_dir = data_dir / "PRE-event"

    # skimage throws an annoying "low contrast warning, so ignore"
    # ignore skimage warnings
    warnings.filterwarnings("ignore")

    label_image_mapping_csvs = list(data_dir.glob("*label_image_mapping.csv"))
    if len(label_image_mapping_csvs) != 1:
        raise RuntimeError

    print(label_image_mapping_csvs[0])
    df = pd.read_csv(label_image_mapping_csvs[0])

    # make output dir
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(exist_ok=True)
    if output_mask_multidim_dir is not None:
        output_mask_multidim_dir = Path(output_mask_multidim_dir)
        assert output_mask_dir != output_mask_multidim_dir
        output_mask_multidim_dir.mkdir(exist_ok=True)

    speed_to_burn_func, channel_value_mult, n_channels, channel_burnValue, append_total_band = get_parameters(
        output_mask_multidim_dir is None, output_conversion_csv
    )

    for label_name, pre_image_name in zip(df["label"], df["pre-event image"]):
        geojson_path = geojson_dir / ("roads_speed_" + label_name)
        image_path = image_dir / pre_image_name

        mask_path_out = output_mask_dir / pre_image_name
        mask_path_out_md = output_mask_multidim_dir / pre_image_name

        speed_mask(
            str(geojson_path),
            str(image_path),
            str(mask_path_out),
            str(mask_path_out_md),
            speed_to_burn_func,
            buffer_distance_meters,
            True,
            channel_value_mult,
            n_channels,
            channel_burnValue,
            append_total_band,
        )


if __name__ == "__main__":
    fire.Fire(main)
