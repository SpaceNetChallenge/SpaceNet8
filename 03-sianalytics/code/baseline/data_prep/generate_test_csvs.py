"""
utility for generating .csv files that can be used with the SN8Dataset pytorch dataset. 

output csvs have the following columns: preimg, postimg, flood, building, road, roadspeed. 
    preimg column contains filepaths to the pre-event image tiles (.tif)
    postimg column contains filepaths to the post-event image tiles (.tif)
    building column contains the filepaths to the binary building labels (.tif)
    road column contains the filepaths to the binary road labels (.tif)
    roadspeed column contains the filepaths to the road speed labels (.tif)
    flood column contains the filepaths to the flood labels (.tif)
"""

import glob
import os
import math
import argparse
import csv

import numpy as np

def write_csv(images, idxs, out_csv_filename):
    print(f"writing out csv: {out_csv_filename}")
    outfile = open(out_csv_filename, "w")
    header = "preimg,postimg,flood,building,road,roadspeed\n"
    outfile.write(header)
    for i in idxs:
        line = images[0][i]
        for j in range(1, len(images)):
            line += ","+images[j][i]
        for j in range(4):
            line+=","
        line+="\n"
        outfile.write(line)
    outfile.close()

def gather_images_masks(image_dir):
    image_types = ["preimg", "postimg"]
    mask_types = ["flood", "building", "road", "roadspeed"]
    images = []
    masks = []
    for i in range(len(image_types)):
        raw_images = glob.glob(os.path.join(image_dir, f"*{image_types[i]}.tif"))
        raw_images.sort()
        images.append(raw_images)
    for i in range(len(mask_types)):
        image_masks = glob.glob(os.path.join(image_dir, f"*{mask_types[i]}.tif"))
        image_masks.sort()
        masks.append(image_masks)
    return images, masks

def make_test_csvs(image_dirs,
                   mapping_csv,
                   out_csv_basename,
                        ):
    pre_images = []
    post_images = []
    annos = []
    with open(mapping_csv) as f:
        print(mapping_csv)
        reader = csv.DictReader(f)
        for r in reader:
            annos.append(r['label'])

    for d in image_dirs:
        pre = glob.glob(os.path.join(d, "PRE-event", "*.tif"))
        post = glob.glob(os.path.join(d, "POST-event", "*.tif"))
        preims, postims = match_im(pre, post, annos)

        post_images.extend(postims)
        pre_images.extend(preims)

    all_images = [[],[]]
    for i in range(len(annos)):
        all_images[0].append(pre_images[i])
        all_images[1].append(post_images[i])

    idxs = np.arange(0, len(all_images[0]))
    #np.random.shuffle(idxs)
    train_idxs = idxs
    print(f"number of images total: {len(all_images[0])}")
    print(f"number of test images: {len(train_idxs)}")
    write_csv(all_images, train_idxs, f"{out_csv_basename}.csv")

def match_im(pre, post, anno):
    out_pre = []
    out_post = []
    for i in anno:
        tileid = os.path.basename(i).split('.')[0]
        pre_im = [j for j in pre if f"_{tileid}.tif" in j][0]
        post_im = [j for j in post if f"_{tileid}.tif" in j][0]
        
        out_pre.append(pre_im)
        out_post.append(post_im)
        
    return out_pre, out_post

def parse_args():
    parser = argparse.ArgumentParser("python3 generate_test_csvs.py --root_dir /nas/Dataset/SpaceNet8/ --aoi_dirs Testing --out_csv sn8_test --out_dir csvs")
    parser.add_argument("--root_dir")
    parser.add_argument("--out_dir",
                       type=str)
    parser.add_argument("--mapping_csv",
                        default='Louisiana-West_Test_Public_label_image_mapping.csv',
                         type=str)
    parser.add_argument("--out_csv_basename",
                         type=str)
    parser.add_argument("--aoi_dirs",
                        type=str,
                        nargs="+")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    root_dir = args.root_dir
    aois = args.aoi_dirs
    out_csv_basename = args.out_csv_basename
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ##### train val split as random
    image_dirs = [os.path.join(root_dir, n) for n in aois]
    make_test_csvs(image_dirs, os.path.join(root_dir, aois[0], args.mapping_csv), os.path.join(out_dir, out_csv_basename))
