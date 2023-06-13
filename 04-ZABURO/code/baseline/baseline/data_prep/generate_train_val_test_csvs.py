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

import numpy as np

def write_csv(images, masks, idxs, out_csv_filename):
    print(f"writing out csv: {out_csv_filename}")
    outfile = open(out_csv_filename, "w")
    header = "preimg,postimg,flood,building,road,roadspeed\n"
    outfile.write(header)
    for i in idxs:
        line = images[0][i]
        for j in range(1, len(images)):
            line += ","+images[j][i]
        for j in range(len(masks)):
            if len(masks[j])!=0:
                line += ","+masks[j][i]
            else:
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

def make_train_val_csvs(image_dirs,
                        out_csv_basename,
                        val_percent=0.15):
    geojsons = []
    pre_images = []
    post_images = []
    build_labels = []
    road_labels = []
    flood_labels = []
    speed_labels = []
    for d in image_dirs:
        anno = glob.glob(os.path.join(d, "annotations", "*.geojson"))
        bldgs = glob.glob(os.path.join(d, "annotations", "masks", "building*.tif"))
        roads = glob.glob(os.path.join(d, "annotations", "masks", "road*.tif"))
        flood = glob.glob(os.path.join(d, "annotations", "masks", "flood*.tif"))
        roadspeed = glob.glob(os.path.join(d, "annotations", "masks", "roadspeed*.tif"))
        pre = glob.glob(os.path.join(d, "PRE-event", "*.tif"))
        post = glob.glob(os.path.join(d, "POST-event", "*.tif"))
        an, bu, ro, fl, rs, preims, postims = match_im_label(anno, bldgs, roads, flood, roadspeed, pre, post)

        geojsons.extend(an)
        build_labels.extend(bu)
        road_labels.extend(ro)
        flood_labels.extend(fl)
        speed_labels.extend(rs)
        post_images.extend(postims)
        pre_images.extend(preims)

    all_images = [[],[]]
    all_masks = [[],[],[],[]]
    for i in range(len(geojsons)):
        all_images[0].append(pre_images[i])
        all_images[1].append(post_images[i])
        all_masks[0].append(flood_labels[i])
        all_masks[1].append(build_labels[i])
        all_masks[2].append(road_labels[i])
        all_masks[3].append(speed_labels[i])

    idxs = np.arange(0, len(all_images[0]))
    np.random.shuffle(idxs)
    n_val = math.ceil(len(idxs)*val_percent)
    if n_val > 0:
        val_idxs = idxs[:n_val+1]
        train_idxs = idxs[n_val+1:]
        print(f"number of images total: {len(all_images[0])}")
        print(f"number of train images: {len(train_idxs)}")
        print(f"number of val images: {len(val_idxs)}")

        write_csv(all_images, all_masks, train_idxs, f"{out_csv_basename}_train.csv")
        write_csv(all_images, all_masks, val_idxs, f"{out_csv_basename}_val.csv")
    else:
        train_idxs = idxs
        print(f"number of images total: {len(all_images[0])}")
        print(f"number of train images: {len(train_idxs)}")
        write_csv(all_images, all_masks, train_idxs, f"{out_csv_basename}.csv")

def match_im_label(anno, bldgs, roads, floods, roadspeeds, pre, post):
    out_pre = []
    out_post = []
    out_anno = []
    out_bu = []
    out_ro = []
    out_fl = []
    out_rs = []
    for i in anno:
        tileid = os.path.basename(i).split('.')[0]
        pre_im = [j for j in pre if f"_{tileid}.tif" in j][0]
        post_im = [j for j in post if f"_{tileid}.tif" in j][0]
        build = [j for j in bldgs if "building_" in j and f"_{tileid}.tif" in j][0]
        road = [j for j in roads if "road_" in j and f"_{tileid}.tif" in j][0]
        flood = [j for j in floods if "flood_" in j and f"_{tileid}.tif" in j][0]
        speed = [j for j in roadspeeds if "roadspeed_" in j and f"_{tileid}.tif" in j][0]
        
        out_anno.append(i)
        out_bu.append(build)
        out_ro.append(road)
        out_fl.append(flood)
        out_rs.append(speed)
        out_pre.append(pre_im)
        out_post.append(post_im)
        
    return out_anno, out_bu, out_ro, out_fl, out_rs, out_pre, out_post

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir")
    parser.add_argument("--out_dir",
                       type=str)
    parser.add_argument("--out_csv_basename",
                         type=str)
    parser.add_argument("--aoi_dirs",
                        type=str,
                        nargs="+")
    parser.add_argument("--val_percent",
                        type=float,
                        required=True,
                        default=0.15)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    root_dir = args.root_dir
    aois = args.aoi_dirs
    out_csv_basename = args.out_csv_basename
    val_percent = args.val_percent
    out_dir = args.out_dir

    ##### train val split as random
    image_dirs = [os.path.join(root_dir, n) for n in aois]
    make_train_val_csvs(image_dirs, os.path.join(out_dir, out_csv_basename), val_percent=val_percent)