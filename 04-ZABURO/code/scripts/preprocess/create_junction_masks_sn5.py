import argparse
import os

from sn8.data_prep.junction_mask import junction_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson_dir", default="", type=str, help="location of geojson labels")
    parser.add_argument("--image_dir", default="", type=str, help="location of geotiffs")
    parser.add_argument("--output_mask_dir", default="", type=str, help="location of output masks")
    parser.add_argument("--road_buffer_meters", default=2.0, type=float)
    parser.add_argument("--junction_buffer_meters", default=8.0, type=float)
    args = parser.parse_args()

    # make output dir
    os.makedirs(args.output_mask_dir, exist_ok=True)

    image_dir = args.image_dir
    geojson_dir = args.geojson_dir
    output_dir = args.output_mask_dir
    images = sorted([z for z in os.listdir(image_dir) if z.endswith(".tif")])
    for j, image_name in enumerate(images):

        image_root = image_name.split(".")[0]
        image_path = os.path.join(image_dir, image_name)

        mask_path_out = os.path.join(output_dir, image_name)

        # Get geojson path
        # SpaceNet chips
        geojson_path = os.path.join(
            geojson_dir,
            image_root.replace("PS-RGB", "geojson_roads_speed").replace("PS-MS", "geojson_roads_speed")
            # geojson_dir, image_root.replace('PS-RGB', 'geojson_roads_speed')
            + ".geojson",
        )

        print(j + 1, "/", len(images), "image:", image_name, "geojson:", geojson_path)

        junction_mask(geojson_path, image_path, mask_path_out, args.road_buffer_meters, args.junction_buffer_meters)
