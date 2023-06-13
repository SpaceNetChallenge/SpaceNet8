import json
import os

import shapely.wkt
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
from PIL import ImageDraw

CLS = {
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4,
}

def create_segmentation(image_filename, label_filename):
    if not image_filename.endswith('.png'):
        return
    if not label_filename.endswith('.json'):
        return

    dst_path = '/data/xView2/'
    scene_name = os.path.splitext(os.path.basename(image_filename))[0]

    image = Image.open(image_filename).convert('RGB')
    label = json.load(open(label_filename))
    features = label['features']
    properties = features['xy']

    size = image.size
    height, width = size
    labelImg = Image.new('L', size, 0)
    drawer = ImageDraw.Draw(labelImg)
    for prop in properties:
        wkt = prop['wkt']
        prop = prop['properties']
        if 'subtype' not in prop:
            continue
        subtype = prop['subtype']
        if subtype in CLS:
            if '_pre_' in image_filename:
                print(subtype)
            wkt = shapely.wkt.loads(wkt)
            polygon = list(zip(*wkt.exterior.coords.xy))
            drawer.polygon(polygon, fill=CLS[subtype])
#       feature_type = prop['feature_type']
#       if feature_type == 'building':
#           wkt = shapely.wkt.loads(wkt)
#           polygon = list(zip(*wkt.exterior.coords.xy))
#           drawer.polygon(polygon, fill=1)

    labelImg = np.array(labelImg, dtype=np.uint8)

    label = Image.fromarray(labelImg).convert('P')
    label.putpalette(np.array([[0, 0, 0], [128, 128, 128], [0, 255, 0], [255, 255, 0], [255, 0, 0]], dtype=np.uint8))
    label_path = os.path.join(dst_path, 'train/cls', f'{scene_name}.png')
    label.save(label_path)

def map_function(data):
    create_segmentation(**data)

def main():

    root = '/data/xView2/train/'
    image_paths = list()
    for root, dirs, files in os.walk('images'):
        for filename in files:
            print(root, dirs, filename)
            if not filename.endswith('.png'):
                continue
            image_path = os.path.join(root, filename)
            image_paths.append(image_path)

    label_paths = list()
    for image_path in image_paths:
        label_path = image_path.replace('images', 'labels')
        label_path = label_path.replace('.png', '.json')
        if not label_path.endswith('.json'):
            continue
        assert os.path.exists(label_path)
        label_paths.append(label_path)

    image_paths = sorted(image_paths)
    label_paths = sorted(label_paths)


    input_args = list()
    for image_path, label_path in zip(image_paths, label_paths):
        input_args.append(
                {'image_filename': image_path,
                 'label_filename': label_path})

    pool = Pool(30)
    for _ in tqdm(pool.imap_unordered(map_function, input_args), total=len(input_args)):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
