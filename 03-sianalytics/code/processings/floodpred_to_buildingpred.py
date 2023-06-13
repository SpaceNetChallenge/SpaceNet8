import os
import glob
import argparse
import rasterio
import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--show-png", action='store_true')
    parser.add_argument("--building", action='store_true')
    parser.add_argument("--road", action='store_true')
    args = parser.parse_args()
    return args


def create_data(filename, rootdir, outdir, classes, show_png, building, road):
    basename = os.path.basename(filename)

    r = rasterio.open(filename)
    meta = r.meta
    r = r.read()[0] # [1, 1300, 1300]

    buildinglabel = np.zeros((r.shape[0], r.shape[1]))
    loadlabel = np.zeros((1, r.shape[0], r.shape[1]))

    if classes == 5:
        nf_building = r == 1
        f_building = r == 2
        out_building = np.logical_or(nf_building, f_building)

        nf_load = r == 3
        f_load = r == 4
        out_road = np.logical_or(nf_load, f_load)

    elif classes == 3:
        out_building = r == 1
        out_road = r == 2

    elif classes == 2 and (building or road):
        foreground = r == 1

        out_building = foreground
        out_road = foreground

    else:
        assert classes == (2 or 3 or 5)
        return None

    buildinglabel[out_building] = 1
    buildinglabel = buildinglabel.astype('uint8')
    loadlabel[:, out_road] = 1
    loadlabel = loadlabel.astype('uint8')
    loadlabel_255 = loadlabel * 254

    loadlabel = np.concatenate([
        np.zeros((3, r.shape[0], r.shape[1])),
        loadlabel_255,
        np.zeros((3, r.shape[0], r.shape[1])),
        loadlabel_255], axis=0)

    if building:
        with rasterio.open(os.path.join(outdir, basename.replace('floodpred.tif', 'buildingpred.tif')), 'w', **meta) as dst:
            buildinglabel_arr = buildinglabel * 255
            dst.write(buildinglabel_arr, 1)

    if road:
        meta.update({'count': 8})
        with rasterio.open(os.path.join(outdir, basename.replace('floodpred.tif', 'roadspeedpred.tif')), 'w', **meta) as dst:
            dst.write(loadlabel)

    if show_png:
        roadlabel = loadlabel_255 == 255 # 254
        buildinglabel[roadlabel[0]] = 2

        img = Image.fromarray(buildinglabel).convert('P')
        img.putpalette(np.array(
                        [[0,0,0], 
                        [255, 255, 255],
                        [0, 0, 255]
                        ], dtype=np.uint8))
        img.save(os.path.join(outdir, basename.replace('.tif', '.png')))


def map_function(data):
    create_data(**data)


def main(rootdir, outdir, classes=3, show_png=False, building=False, road=False):
    assert classes == (2 or 3 or 5)

    os.makedirs(outdir, exist_ok=True)
    filelist = glob.glob(rootdir + '/*_floodpred.tif')

    if building:
        log = "building"
    elif road:
        log = "road"
    else:
        log = "flood"
    print("Number of the predicted files: {}".format(len(filelist)))
    print(f"Convert floodpred to {log}pred")

    input_args = []
    for filename in filelist:
        arg = dict()
        arg['filename'] = filename
        arg['rootdir'] = rootdir
        arg['outdir'] = outdir
        arg['classes'] = classes
        arg['show_png'] = show_png
        arg['building'] = building
        arg['road'] = road
        input_args.append(arg)

    pool = Pool(20)
    for _ in tqdm(pool.imap_unordered(map_function, input_args), total=len(input_args)):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
