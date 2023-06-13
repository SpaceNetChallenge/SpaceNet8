from pathlib import Path

import fire
import pandas as pd
from somen.types import PathLike

from sn8.data_prep.junction_mask import junction_mask


def main(
    data_dir: PathLike,
    output_mask_dir: PathLike,
    road_buffer_meters: float = 2.0,
    junction_buffer_meters: float = 8.0,
) -> None:

    data_dir = Path(data_dir)
    geojson_dir = data_dir / "annotations" / "prepped_cleaned"
    image_dir = data_dir / "PRE-event"

    label_image_mapping_csvs = list(data_dir.glob("*label_image_mapping.csv"))
    if len(label_image_mapping_csvs) != 1:
        raise RuntimeError

    print(label_image_mapping_csvs[0])
    df = pd.read_csv(label_image_mapping_csvs[0])

    # make output dir
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(exist_ok=True)

    for label_name, pre_image_name in zip(df["label"], df["pre-event image"]):
        geojson_path = geojson_dir / ("roads_speed_" + label_name)
        image_path = image_dir / pre_image_name
        mask_path_out = output_mask_dir / pre_image_name

        junction_mask(
            str(geojson_path), str(image_path), str(mask_path_out), road_buffer_meters, junction_buffer_meters
        )


if __name__ == "__main__":
    fire.Fire(main)
