import sys
import os
import glob


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    g = glob.glob(in_dir+'/*.tif')
    for item in g:
        cmd = f'gdalwarp  -r bilinear -ot Byte -of GTiff -ts 1300 1300 {item} {os.path.join(out_dir, os.path.basename(item))}'
        os.system(cmd)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
