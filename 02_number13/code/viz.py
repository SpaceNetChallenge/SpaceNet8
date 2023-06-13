import numpy as np
from skimage import io
import glob
import os
import sys
import glob
import cv2
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm


def main(dir_pre, dir):
    for item in glob.glob(dir + '/flood_*'):
        plt.figure(figsize=(10, 10))
        build_all = io.imread(item.replace('flood_', 'building_'))
        road_all = io.imread(item.replace('flood_', 'road_'))
        ball = cv2.merge(
            (build_all[:, :, np.newaxis], road_all[:, :, np.newaxis], np.zeros((1300, 1300, 1), dtype=np.uint8)))
        plt.subplot(131)
        img = io.imread(os.path.join(dir_pre, os.path.basename(item).replace('.png', '.tif').replace('flood_', '')))
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(ball)
        plt.subplot(133)
        plt.imshow(io.imread(item))
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
