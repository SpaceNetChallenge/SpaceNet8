import argparse
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm

# isort: off
from spacenet8_model.utils.postproc_road_to_vector import make_skeleton, img_to_ske_G, build_wkt_linestrings
# isort: on


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foundation', required=True)
    parser.add_argument('--road_thresh',
                        type=float,
                        default=0.3)
    parser.add_argument('--min_spur_length_pix',
                        type=int,
                        default=20)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


def postprocess(foundation_path, args, aoi):
    # TODO:
    road_channels = [3, 4]

    assert os.path.exists(foundation_path), foundation_path

    img, ske = make_skeleton(
        foundation_path,
        thresh=0.3,
        debug=False,  # XXX: debug is not used
        fix_borders=False,
        replicate=5,
        clip=2,
        img_mult=255,
        hole_size=300,
        cv2_kernel_close=7,
        cv2_kernel_open=7,
        use_medial_axis=False,
        skelton_bands=road_channels,
        road_thresh=args.road_thresh,
        verbose=False
    )

    if ske is None:
        return None  # no road in the image

    G, ske, img_refine = img_to_ske_G(
        img,
        ske,
        min_spur_length_pix=args.min_spur_length_pix,
        out_gpickle='',
        verbose=False)

    out_csv = None
    out_shp = None
    df = build_wkt_linestrings(
        G,
        out_csv,
        out_shp,
        reference_image_filename=foundation_path,
        add_small=True,
        verbose=False,
        super_verbose=False)

    return df


def process_aoi(args, aoi):
    foundation_paths = glob(os.path.join(args.foundation, aoi, '*.tif'))
    foundation_paths.sort()

    df = pd.DataFrame(columns=['ImageId', 'WKT_Pix'])
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(foundation_paths)) as pbar:
            for ret in pool.imap_unordered(partial(postprocess, args=args, aoi=aoi), foundation_paths):
                if ret is not None:
                    df = df.append(ret)
                pbar.update()
    return df


def main():
    args = parse_args()

    aois = [d for d in os.listdir(args.foundation) if os.path.isdir(os.path.join(args.foundation, d))]
    df = pd.DataFrame(columns=['ImageId', 'WKT_Pix'])
    for aoi in aois:
        print(f'processing {aoi} AOI')
        ret = process_aoi(args, aoi)
        df = df.append(ret)

    print(df.head(15))

    out_dir = os.path.basename(os.path.normpath(args.foundation))
    if args.val:
        out_dir = os.path.join(args.artifact_dir, '_val/road_vectors', out_dir)
    else:
        out_dir = os.path.join(args.artifact_dir, 'road_vectors', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'road_vectors.csv')
    df.to_csv(out_path, index=False)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
