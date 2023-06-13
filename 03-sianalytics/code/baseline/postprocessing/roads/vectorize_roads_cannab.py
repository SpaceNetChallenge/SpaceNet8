import os
from os import path, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import timeit
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import sys
from shapely.geometry.linestring import LineString
from skimage.morphology import skeletonize_3d, square, erosion, dilation, medial_axis
from skimage.measure import label, regionprops, approximate_polygon
from math import hypot, sin, cos, asin, acos, radians
from sklearn.neighbors import KDTree
from shapely.wkt import dumps
from osgeo import gdal
import glob
import argparse
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',
                        help='a directory full of predictions masks (.tif)',
                        required=False,
                        default="",
                        type=str)
    parser.add_argument('--outdir',
                        help='the output directory to write output to',
                        required=False,
                        default="",
                        type=str)
    args = parser.parse_args()
    return args


def get_ordered_coords(lbl, l, coords):
    res = []
    
    nxt = coords[0]
    for i in range(coords.shape[0]):
        y, x = coords[i]
        cnt = 0
        for y0 in range(max(0, y-1), min(1300, y+2)):
            for x0 in range(max(0, x-1), min(1300, x+2)):
                if lbl[y0, x0] == l:
                    cnt += 1
        if cnt == 2:
            nxt = coords[i]
            lbl[y, x] = 0
            res.append(nxt)
            break
    while nxt is not None:
        y, x = nxt
        fl = False
        for y0 in range(max(0, y-1), min(1300, y+2)):
            for x0 in range(max(0, x-1), min(1300, x+2)):
                if lbl[y0, x0] == l:
                    fl = True
                    nxt = np.asarray([y0, x0])
                    lbl[y0, x0] = 0
                    res.append(nxt)
                    break
            if fl:
                break
        if not fl:
            nxt = None
    return np.asarray(res)
    
def point_hash(x, y):
    x_int = int(x)
    y_int = int(y)
    h = x_int * 10000 + y_int
    return h

def pair_hash(p1, p2):
    h1 = point_hash(p1[0], p1[1])
    h2 = point_hash(p2[0], p2[1])
    return np.int64(h1 * 1e8 + h2)

def get_next_point(p1, p2, add_dist):
    l = hypot(p1[0] - p2[0], p1[1] - p2[1])
    dx = (p2[1] - p1[1]) / l
    dy = (p2[0] - p1[0]) / l
    x3 = int(round(p1[1] + (l + add_dist) * dx))
    y3 = int(round(p1[0] + (l + add_dist) * dy))
    if x3 < 0:
        x3 = 0
    if y3 < 0:
        y3 = 0
    if x3 > 1299:
        x3 = 1299
    if y3 > 1299:
        y3 = 1299
    return np.asarray([y3, x3])
   
def try_connect(p1, p2, a, max_dist, road_msk, min_prob, msk, roads, r=3):
    prob = []
    r_id = road_msk[p2[0], p2[1]]
    hashes = set([point_hash(p2[1], p2[0])])
    l = hypot(p1[0] - p2[0], p1[1] - p2[1])
    dx = (p2[1] - p1[1]) / l
    dy = (p2[0] - p1[0]) / l
    if a != 0:
        l = 0
        p1 = p2
        _a = asin(dy) + a
        dy = sin(_a)
        _a = acos(dx) + a
        dx = cos(_a)
    step = 0.5
    d = step
    while d < max_dist:
        x3 = int(round(p1[1] + (l + d) * dx))
        y3 = int(round(p1[0] + (l + d) * dy))
        if x3 < 0 or y3 < 0 or x3 > 1299 or y3 > 1299:
            return None
        h = point_hash(x3, y3)
        if h not in hashes:
            prob.append(msk[y3, x3])
            hashes.add(h)
            for x0 in range(max(0, x3 - r), min(1299, x3 + r + 1)):
                for y0 in range(max(0, y3 - r), min(1299, y3 + r + 1)):
                    if road_msk[y0, x0] > 0 and road_msk[y0, x0] != r_id:
                        p3 = np.asarray([y0, x0])
                        r2_id = road_msk[y0, x0] - 1
                        t = KDTree(roads[r2_id])
                        clst = t.query(p3[np.newaxis, :])
                        if clst[0][0][0] < 10:
                            p3 = roads[r2_id][clst[1][0][0]]
                        if np.asarray(prob).mean() > min_prob:
                            return p3
                        else:
                            return None
        d += step
    return None

def inject_point(road, p):
    for i in range(road.shape[0]):
        if road[i, 0] == p[0] and road[i, 1] == p[1]:
            return road, []
    new_road = []
    to_add = True
    new_hashes = []
    for i in range(road.shape[0] - 1):
        new_road.append(road[i])
        if (to_add and min(road[i, 0], road[i+1, 0]) <= p[0] 
            and max(road[i, 0], road[i+1, 0]) >= p[0]
            and min(road[i, 1], road[i+1, 1]) <= p[1]
            and max(road[i, 1], road[i+1, 1]) >= p[1]):
            to_add = False
            new_road.append(p)
            new_hashes.append(pair_hash(p, road[i]))
            new_hashes.append(pair_hash(road[i], p))
            new_hashes.append(pair_hash(p, road[i+1]))
            new_hashes.append(pair_hash(road[i+1], p))
    new_road.append(road[-1])
    return np.asarray(new_road), new_hashes

def process_file(filename):
    res_rows = []
    # msk = msk.astype('uint8')
    # if save_to is not None:
    #     cv2.imwrite(save_to, msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # msk = rasterio.open(filename).read()[7,:,:]
    msk = gdal.Open(filename).ReadAsArray()[7]
    msk2 = np.lib.pad(msk, ((22, 22), (22, 22)), 'symmetric')

    par2 = np.asarray([  3,   0,   5,  10, 120, 120,  24,   4,   9])
    par = np.asarray([  1,   2,   0,   0,   2,   1,   1,   0,   1,   5,   2,   2, 168])

    thr = par[12]

    msk2 = 1 * (msk2 > thr)
    msk2 = msk2.astype(np.uint8)

    if par2[0] > 0:
        msk2 = dilation(msk2, square(par2[0]))
    if par2[1] > 0:
        msk2 = erosion(msk2, square(par2[1]))

    skeleton = skeletonize_3d(msk2)
    skeleton = skeleton[22:1322, 22:1322]

    lbl0 = label(skeleton)
    props0 = regionprops(lbl0)

    cnt = 0
    crosses = []
    for x in range(1300):
        for y in range(1300):
            # skeleton이 존재한다는 뜻
            if skeleton[y, x] == 1:
                # 교차점이 3개 이상인 지점 즉 삼거리 이상을 별도의 crosses포인트로 저장
                if skeleton[max(0, y-1):min(1300, y+2), max(0, x-1):min(1300, x+2)].sum() > 3:
                    cnt += 1
                    crss = []
                    crss.append((x, y))
                    for y0 in range(max(0, y-1), min(1300, y+2)):
                        for x0 in range(max(0, x-1), min(1300, x+2)):
                            if x == x0 and y == y0:
                                continue
                            if skeleton[max(0, y0-1):min(1300, y0+2), max(0, x0-1):min(1300, x0+2)].sum() > 3:
                                crss.append((x0, y0))
                    crosses.append(crss)
                    
    # hash값으로 변경
    cross_hashes = []
    for crss in crosses:
        crss_hash = set([])
        for x0, y0 in crss:
            crss_hash.add(point_hash(x0, y0))
            skeleton[y0, x0] = 0
        cross_hashes.append(crss_hash)

    new_crosses = []
    i = 0
    while i < len(crosses):
        new_hashes = set([])
        new_hashes.update(cross_hashes[i])
        new_crss = crosses[i][:]
        fl = True
        while fl:
            fl = False
            j = i + 1
            while j < len(crosses):
                if len(new_hashes.intersection(cross_hashes[j])) > 0:
                    new_hashes.update(cross_hashes[j])
                    new_crss.extend(crosses[j])
                    cross_hashes.pop(j)
                    crosses.pop(j)
                    fl = True
                    break
                j += 1
        mean_p = np.asarray(new_crss).mean(axis=0).astype('int')
        if len(new_crss) > 1:
            t = KDTree(new_crss)
            mean_p = new_crss[t.query(mean_p[np.newaxis, :])[1][0][0]]
        new_crosses.append([(mean_p[0], mean_p[1])] + new_crss)
        i += 1
    
    crosses = new_crosses

    lbl = label(skeleton)
    props = regionprops(lbl)

    connected_roads = []
    connected_crosses = [set([]) for p in props]
    for i in range(len(crosses)):
        rds = set([])
        for j in range(len(crosses[i])):
            x, y = crosses[i][j]
            for y0 in range(max(0, y-1), min(1300, y+2)):
                for x0 in range(max(0, x-1), min(1300, x+2)):
                    if lbl[y0, x0] > 0:
                        rds.add(lbl[y0, x0])
                        connected_crosses[lbl[y0, x0]-1].add(i)
        connected_roads.append(rds)


    res_roads = []

    tot_dist_min = par2[2]
    coords_min = par2[3]

    for i in range(len(props)):
        coords = props[i].coords
        crss = list(connected_crosses[i])
        tot_dist = props0[lbl0[coords[0][0], coords[0][1]]-1].area

        if (tot_dist < tot_dist_min) or (coords.shape[0] < coords_min and len(crss) < 2):
            continue
        if coords.shape[0] == 1:
            coords = np.asarray([coords[0], coords[0]])
        else:
            coords = get_ordered_coords(lbl, i+1, coords)
        for j in range(len(crss)):
            x, y = crosses[crss[j]][0]
            d1 = abs(coords[0][0] - y) + abs(coords[0][1] - x)
            d2 = abs(coords[-1][0] - y) + abs(coords[-1][1] - x)
            if d1 < d2:
                coords[0][0] = y
                coords[0][1] = x
            else:
                coords[-1][0] = y
                coords[-1][1] = x
        coords_approx = approximate_polygon(coords, 1.5)
        res_roads.append(coords_approx)

    hashes = set([])
    final_res_roads = []
    for r in res_roads:
        if r.shape[0] > 2:
            final_res_roads.append(r)
            for i in range(1, r.shape[0]):
                p1 = r[i-1]
                p2 = r[i]
                h1 = pair_hash(p1, p2)
                h2 = pair_hash(p2, p1)
                hashes.add(h1)
                hashes.add(h2)

    for r in res_roads:
        if r.shape[0] == 2:
            p1 = r[0]
            p2 = r[1]
            h1 = pair_hash(p1, p2)
            h2 = pair_hash(p2, p1)
            if not (h1 in hashes or h2 in hashes):
                final_res_roads.append(r)
                hashes.add(h1)
                hashes.add(h2)

    end_points = {}
    for r in res_roads:
        h = point_hash(r[0, 0], r[0, 1])
        if not (h in end_points.keys()):
            end_points[h] = 0
        end_points[h] = end_points[h] + 1
        h = point_hash(r[-1, 0], r[-1, 1])
        if not (h in end_points.keys()):
            end_points[h] = 0
        end_points[h] = end_points[h] + 1

    road_msk = np.zeros((1300, 1300), dtype=np.int32)
    road_msk = road_msk.copy()
    thickness = 1
    for j in range(len(final_res_roads)):
        l = final_res_roads[j]
        for i in range(len(l) - 1):
            cv2.line(road_msk, (int(l[i, 1]), int(l[i, 0])), (int(l[i+1, 1]), int(l[i+1, 0])), j+1, thickness)

    connect_dist = par2[4]

    min_prob = par2[5]
    angles_to_check = [0, radians(5), radians(-5), radians(10), radians(-10), radians(15), radians(-15)]

    add_dist = par2[6]
    add_dist2 = par2[7]

    con_r = par2[8]

    for i in range(len(final_res_roads)):
        h = point_hash(final_res_roads[i][0, 0], final_res_roads[i][0, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][1]
            p2 = final_res_roads[i][0]            
            p3 = try_connect(p1, p2, 0, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)          
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((p3, final_res_roads[i]))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2
        h = point_hash(final_res_roads[i][-1, 0], final_res_roads[i][-1, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][-2]
            p2 = final_res_roads[i][-1]
            p3 = try_connect(p1, p2, 0, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((final_res_roads[i], p3))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2

    for i in range(len(final_res_roads)):
        h = point_hash(final_res_roads[i][0, 0], final_res_roads[i][0, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][1]
            p2 = final_res_roads[i][0]
            p3 = None
            for a in angles_to_check:
                p3 = try_connect(p1, p2, a, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
                if p3 is not None:
                    break
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)          
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((p3, final_res_roads[i]))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2
            else:
                p3 = get_next_point(p1, p2, add_dist)
                if not (p3[0] < 2 or p3[1] < 2 or p3[0] > 1297 or p3[1] > 1297):
                    p3 = get_next_point(p1, p2, add_dist2)
                if (p3[0] != p2[0] or p3[1] != p2[1]) and (road_msk[p3[0], p3[1]] == 0):
                    h1 = pair_hash(p2, p3)
                    h2 = pair_hash(p3, p2)
                    if not (h1 in hashes or h2 in hashes):
                        final_res_roads[i] = np.vstack((p3, final_res_roads[i]))
                        hashes.add(h1)
                        hashes.add(h2)
                        tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                        tmp_road_msk = tmp_road_msk.copy()
                        cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                        road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                        road_msk = road_msk.copy()
                        end_points[point_hash(p3[0], p3[1])] = 2

        h = point_hash(final_res_roads[i][-1, 0], final_res_roads[i][-1, 1])
        if end_points[h] == 1:
            p1 = final_res_roads[i][-2]
            p2 = final_res_roads[i][-1]
            p3 = None
            for a in angles_to_check:
                p3 = try_connect(p1, p2, a, connect_dist, road_msk, min_prob, msk, final_res_roads, con_r)
                if p3 is not None:
                    break
            if p3 is not None:
                h1 = pair_hash(p2, p3)
                h2 = pair_hash(p3, p2)
                if not (h1 in hashes or h2 in hashes):
                    r_id = road_msk[p3[0], p3[1]] - 1
                    final_res_roads[r_id], new_hashes = inject_point(final_res_roads[r_id], p3)
                    hashes.update(new_hashes)
                    tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                    tmp_road_msk = tmp_road_msk.copy()
                    cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                    road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                    road_msk = road_msk.copy()
                    final_res_roads[i] = np.vstack((final_res_roads[i], p3))
                    hashes.add(h1)
                    hashes.add(h2)
                    end_points[point_hash(p3[0], p3[1])] = 2
            else:
                p3 = get_next_point(p1, p2, add_dist)
                if not (p3[0] < 2 or p3[1] < 2 or p3[0] > 1297 or p3[1] > 1297):
                    p3 = get_next_point(p1, p2, add_dist2)
                if (p3[0] != p2[0] or p3[1] != p2[1]) and (road_msk[p3[0], p3[1]] == 0):
                    h1 = pair_hash(p2, p3)
                    h2 = pair_hash(p3, p2)
                    if not (h1 in hashes or h2 in hashes):
                        final_res_roads[i] = np.vstack((final_res_roads[i], p3))
                        hashes.add(h1)
                        hashes.add(h2)
                        tmp_road_msk = np.zeros((1300, 1300), dtype=np.int32)
                        tmp_road_msk = tmp_road_msk.copy()
                        cv2.line(tmp_road_msk, (p2[1], p2[0]), (p3[1], p3[0]), i+1, thickness)
                        road_msk[road_msk == 0] = tmp_road_msk[road_msk == 0]
                        road_msk = road_msk.copy()
                        end_points[point_hash(p3[0], p3[1])] = 2

    lines = [LineString(r[:, ::-1]) for r in final_res_roads]

    if len(lines) == 0:
        pass
    else:
        for l in lines:
            res_rows.append({'ImageId': filename, 'WKT_Pix': dumps(l, rounding_precision=0)})

    return res_rows

def main(rootdir, outdir):
    img_list = glob.glob(rootdir + '/*_roadspeedpred.tif')
    img_list = sorted(img_list)
    print("Vectorize Roads by Cannab's Method in SN3 solution")
    print("Nums of _roadspeedpred.tif images: {}".format(len(img_list)))

    res = []
    pool = Pool(20)
    for r in tqdm(pool.imap_unordered(process_file, img_list), total=len(img_list)):
        res.extend(r)

    df = pd.DataFrame.from_dict(res)

    print(df.head(5))
    print(len(df))

    df.to_csv(os.path.join(outdir, 'sknw_wkt.csv'), index=False)
    print("Done! Vectorizing Roads is finished")


if __name__ == '__main__':
    # rootdir = '/data/workspace/doyoungi/SpaceNet8/output/submit_0820_2/foundation'
    args = parse_args()
    main(**vars(args))
    