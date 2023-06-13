import argparse
import os
import warnings

from sn8.data_prep.speed_masks import get_parameters, speed_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson_dir", default="", type=str, help="location of geojson labels")
    parser.add_argument("--image_dir", default="", type=str, help="location of geotiffs")
    parser.add_argument("--output_conversion_csv", default="", type=str, help="location of output conversion file")
    parser.add_argument("--output_mask_dir", default="", type=str, help="location of output masks")
    parser.add_argument(
        "--output_mask_multidim_dir",
        default="",
        type=str,
        help="location of output masks for binned case " " set to " " to use continuous case",
    )
    parser.add_argument("--buffer_distance_meters", default=2, type=int, help="width of road buffer in meters")
    args = parser.parse_args()

    # hardcoded for now...
    buffer_distance_meters = args.buffer_distance_meters
    verbose = True
    # resave_pkl = False  # True

    # skimage throws an annoying "low contrast warning, so ignore"
    # ignore skimage warnings
    warnings.filterwarnings("ignore")

    # make output dir
    os.makedirs(args.output_mask_dir, exist_ok=True)
    if len(args.output_mask_multidim_dir) != 0:
        os.makedirs(args.output_mask_multidim_dir, exist_ok=True)

    speed_to_burn_func, channel_value_mult, n_channels, channel_burnValue, append_total_band = get_parameters(
        len(args.output_mask_multidim_dir) == 0, args.output_conversion_csv
    )

    ###########################################################################
    image_dir = args.image_dir
    geojson_dir = args.geojson_dir
    output_dir = args.output_mask_dir
    output_dir_multidim = args.output_mask_multidim_dir
    images = sorted([z for z in os.listdir(image_dir) if z.endswith(".tif")])
    for j, image_name in enumerate(images):

        image_root = image_name.split(".")[0]
        # image_root = image_name.split('RGB-PanSharpen_')[-1].split('.')[0]
        image_path = os.path.join(image_dir, image_name)

        mask_path_out = os.path.join(output_dir, image_name)
        mask_path_out_md = os.path.join(output_dir_multidim, image_name) if output_dir_multidim != "" else ""

        # Get geojson path
        # SpaceNet chips
        geojson_path = os.path.join(
            geojson_dir,
            image_root.replace("PS-RGB", "geojson_roads_speed").replace("PS-MS", "geojson_roads_speed")
            # geojson_dir, image_root.replace('PS-RGB', 'geojson_roads_speed')
            + ".geojson",
        )
        # # Contiguous files
        # geojson_path = os.path.join(geojson_dir, image_root + '.geojson')
        # if (j % 100) == 0:
        if (j % 1) == 0:
            print(j + 1, "/", len(images), "image:", image_name, "geojson:", geojson_path)
        if j > 0:
            verbose = False

        speed_mask(
            geojson_path,
            image_path,
            mask_path_out,
            mask_path_out_md,
            speed_to_burn_func,
            buffer_distance_meters,
            verbose,
            channel_value_mult,
            n_channels,
            channel_burnValue,
            append_total_band,
        )


###############################################################################
if __name__ == "__main__":
    main()
