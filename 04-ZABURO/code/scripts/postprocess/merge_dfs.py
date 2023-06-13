from pathlib import Path

import fire
import pandas as pd
from somen.types import PathLike


def main(data_dir: PathLike, road_df_dir: PathLike, building_df_dir: PathLike, output_path: PathLike) -> None:
    data_dir = Path(data_dir)
    road_df_dir = Path(road_df_dir)
    building_df_dir = Path(building_df_dir)

    label_image_mapping_csvs = list(data_dir.glob("*label_image_mapping.csv"))
    if len(label_image_mapping_csvs) != 1:
        raise RuntimeError

    print(label_image_mapping_csvs[0])
    df = pd.read_csv(label_image_mapping_csvs[0])

    dfs = []
    for pre_name in df["pre-event image"]:
        image_path = data_dir / "PRE-event" / pre_name
        df_road = pd.read_csv(road_df_dir / (image_path.stem + "_wkt_df.csv"))
        df_road["ImageId"] = image_path.stem
        df_bldg = pd.read_csv(building_df_dir / (image_path.stem + "_wkt_df.csv"))
        df_bldg["ImageId"] = image_path.stem
        dfs.append(df_road)
        dfs.append(df_bldg)
    df = pd.concat(dfs, axis=0)
    del dfs

    columns = ["ImageId", "Object", "WKT_Pix", "Flooded", "length_m", "travel_time_s"]
    df[columns].fillna(0.0).to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
