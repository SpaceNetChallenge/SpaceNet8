import argparse
import math
import os
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from shapely.wkt import dumps, loads
from skimage.morphology import dilation, square
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True)
    parser.add_argument('--flood', required=True)
    parser.add_argument('--flood_thresh', type=float, default=0.3)
    parser.add_argument('--flood_area_ratio', type=float, default=0.3)
    parser.add_argument('--flood_dilation_ks', type=int, default=5)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


pre_image_blacklist = [
    # Foundation error:
    # - Louisiana-East_Training_Public
    '10300100AF395C00_2_18_35.tif',  # building FN
    '10300100AF395C00_2_19_35.tif',  # building FN
    '10400100684A4B00_1_22_70.tif',  # building FN
    '10400100684A4B00_1_23_70.tif',  # building FN
    '10400100684A4B00_1_24_70.tif',  # building FN
    '10400100684A4B00_1_25_70.tif',  # building FN
    '10400100684A4B00_1_26_70.tif',  # building FN
    '10400100684A4B00_1_2_84.tif',  # building FN
    # Flood error:
    # - Germany_Training_Public
    '10500500C4DD7000_0_26_62.tif',  # warping
    '10500500C4DD7000_0_27_62.tif',  # warping
    '10500500C4DD7000_0_27_63.tif',  # flood road FP
    '10500500C4DD7000_0_27_64.tif',  # flood road FP
    '10500500C4DD7000_0_29_70.tif',  # warping
    '10500500C4DD7000_0_30_70.tif',  # warping
    # - Louisiana-East_Training_Public
    '10300100AF395C00_2_13_45.tif',  # flood road & building FN
    '10300100AF395C00_2_13_46.tif',  # flood building FN
    '10300100AF395C00_2_13_47.tif',  # flood road & building FN
    '10300100AF395C00_2_14_46.tif',  # flood building FN
    '10300100AF395C00_2_22_43.tif',  # flood road & building FN
    '105001001A0FFC00_0_12_13.tif',  # flood road FN
    '105001001A0FFC00_0_16_14.tif',  # flood road FN
    '105001001A0FFC00_0_17_15.tif',  # flood road FN
    '105001001A0FFC00_0_20_17.tif',  # flood road & building FN
    '10400100684A4B00_1_15_88.tif',  # flood road FN
    '10400100684A4B00_1_15_93.tif',  # flood road FN
    '10400100684A4B00_1_16_73.tif',  # flood road FN
    '10400100684A4B00_1_20_82.tif',  # flood building FN
    '10400100684A4B00_1_21_79.tif',  # flood building FN
    '10400100684A4B00_1_21_86.tif',  # flood building FN
    '10400100684A4B00_1_22_79.tif',  # flood building FN
    '10400100684A4B00_1_23_78.tif',  # flood road & building FN
    '10400100684A4B00_1_23_79.tif',  # flood road & building FN
]


def write_road_submission_shapefile(df, out_shapefile):
    df = df.reset_index()  # make sure indexes pair with number of rows

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    # Create the output shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(out_shapefile)
    out_layer = ds.CreateLayer(out_shapefile[:-4], srs, ogr.wkbLineString)
    
    fieldnames = ['ImageId', 'Object', 'Flooded', 'WKT_Pix', 'WKT_Geo', 'length_m']
    
    field_name = ogr.FieldDefn('ImageId', ogr.OFTString)
    field_name.SetWidth(100)
    out_layer.CreateField(field_name)
    ob = ogr.FieldDefn('Object', ogr.OFTString)
    ob.SetWidth(10)
    out_layer.CreateField(ob)
    flooded = ogr.FieldDefn('Flooded', ogr.OFTString)
    flooded.SetWidth(5)
    out_layer.CreateField(flooded)
    pix = ogr.FieldDefn('WKT_Pix', ogr.OFTString)
    pix.SetWidth(255)
    out_layer.CreateField(pix)
    geo = ogr.FieldDefn('WKT_Geo', ogr.OFTString)
    geo.SetWidth(255)
    out_layer.CreateField(geo)
    out_layer.CreateField(ogr.FieldDefn('length_m', ogr.OFTReal))
    #out_layer.CreateField(ogr.FieldDefn('travel_t_s', ogr.OFTReal))

    # Create the feature and set values
    featureDefn = out_layer.GetLayerDefn()
    
    for index, row in df.iterrows():
        
        outFeature = ogr.Feature(featureDefn)
        for j in fieldnames:
            if j == "travel_time_s":
                pass
                #outFeature.SetField('travel_t_s', row[j])
            else:
                outFeature.SetField(j, row[j])
        
        geom = ogr.CreateGeometryFromWkt(row["WKT_Geo"])
        outFeature.SetGeometry(geom)
        out_layer.CreateFeature(outFeature)
        outFeature = None
    ds = None


def pkl_dir_to_wkt(pkl_dir,
                   weight_keys=['length', 'travel_time_s'],
                   verbose=False):
    """
    Create submission wkt from directory full of graph pickles
    """
    wkt_list = []

    pkl_list = sorted([z for z in os.listdir(pkl_dir) if z.endswith('.gpickle')])
    for i, pkl_name in enumerate(tqdm(pkl_list)):
        # print(pkl_name)
        G = nx.read_gpickle(os.path.join(pkl_dir, pkl_name))
        
        # ensure an undirected graph
        if verbose:
            print(i, "/", len(pkl_list), "num G.nodes:", len(G.nodes()))

        #name_root = pkl_name.replace('PS-RGB_', '').replace('PS-MS_', '').split('.')[0]
        name_root = pkl_name.replace("_roadspeedpred", '').split('.')[0]

        # AOI_root = 'AOI' + pkl_name.split('AOI')[-1]
        # name_root = AOI_root.split('.')[0].replace('PS-RGB_', '')
        if verbose:
            print("name_root:", name_root)
        
        # if empty, still add to submission
        if len(G.nodes()) == 0:
            wkt_item_root = [name_root, 'Road', 'LINESTRING EMPTY', 'LINESTRING EMPTY', 'False']
            if len(weight_keys) > 0:
                weights = ['Null' for w in weight_keys]
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

        # extract geometry pix wkt, save to list
        seen_edges = set([])
        for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):
            # make sure we haven't already seen this edge
            if (u, v) in seen_edges or (v, u) in seen_edges:
                if verbose:
                    print(u, v, "already catalogued!")
                continue
            else:
                seen_edges.add((u, v))
                seen_edges.add((v, u))
            geom_pix = attr_dict['geometry_pix']
            if type(geom_pix) != str:
                geom_pix_wkt = attr_dict['geometry_pix'].wkt
            else:
                geom_pix_wkt = geom_pix
            
            geom_geo = attr_dict["geometry_utm_wkt"].wkt
            #if type(geom_geo) != str:
            #    geom_geo_wkt = attr_dict["geometry_wkt"].wkt
            #else:
            #    geom_geo_wkt = geom_geo
            # geometry_wkt is in a UTM coordinate system..
            geom = ogr.CreateGeometryFromWkt(geom_geo)

            targetsrs = osr.SpatialReference()
            targetsrs.ImportFromEPSG(4326)

            utm_zone = attr_dict['utm_zone']
            source = osr.SpatialReference() # the input dataset is in wgs84
            source.ImportFromProj4(f'+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs')
            transform_to_utm = osr.CoordinateTransformation(source, targetsrs)
            geom.Transform(transform_to_utm)
            geom_geo_wkt = geom.ExportToWkt()

            # check edge lnegth
            #if attr_dict['length'] > 5000:
            #    print("Edge too long!, u,v,data:", u,v,attr_dict)
            #    return
            
            if verbose:
                print(i, "/", len(G.edges()), "u, v:", u, v)
                print("  attr_dict:", attr_dict)
                print("  geom_pix_wkt:", geom_pix_wkt)
                print("  geom_geo_wkt:", geom_geo_wkt)

            wkt_item_root = [name_root, 'Road', geom_pix_wkt, geom_geo_wkt, 'False']
            if len(weight_keys) > 0:
                weights = [attr_dict[w] for w in weight_keys]
                if verbose:
                    print("  weights:", weights)
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

    if verbose:
        print("wkt_list:", wkt_list)

    # create dataframe
    if len(weight_keys) > 0:
        cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded'] + weight_keys
    else:
        cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded']

    # use 'length_m' and 'travel_time_s' instead?
    cols_new = []
    for z in cols:
        if z == 'length':
            cols_new.append('length_m')
        elif z == 'travel_time':
            cols_new.append('travel_time_s')
        else:
            cols_new.append(z)
    cols = cols_new
    # cols = [z.replace('length', 'length_m') for z in cols]
    # cols = [z.replace('travel_time', 'travel_time_s') for z in cols]
    # print("cols:", cols)

    df = pd.DataFrame(wkt_list, columns=cols)
    # print("df:", df)
    # save
    #if len(output_csv_path) > 0:
    #    df.to_csv(output_csv_path, index=False)
    return df


def insert_flood_pred(flood_pred_dir, df, road_flood_channel, flood_thresh, flood_area_ratio, flood_dilation_ks):
    flood_road_label = 4  # same as sn-8 baseline (any positive int should be okay)

    dy=2
    dx=2
    cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded', 'length_m']
    out_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        imageid = row["ImageId"]
        
        flood_pred_filename = os.path.join(flood_pred_dir, f"{imageid}.tif")
        assert(os.path.exists(flood_pred_filename)), "flood prediction file for this linestring doesn't exist"

        ds = gdal.Open(flood_pred_filename)
        nrows = ds.RasterYSize
        ncols = ds.RasterXSize

        # XXX: kimura modified sn-8 baseline
        flood_mask = ds.ReadAsArray()[road_flood_channel].astype(float) / 255.0
        flood_arr = np.zeros(shape=flood_mask.shape, dtype=np.uint8)
        flood_arr[flood_mask >= flood_thresh] = flood_road_label  # 4: flooded road, 0: others
        if flood_dilation_ks > 0:
            flood_arr = dilation(flood_arr, square(flood_dilation_ks))  # dilate to cope with alignment error b/w pre and post images

        if row["WKT_Pix"] != "LINESTRING EMPTY":
            geom = ogr.CreateGeometryFromWkt(row["WKT_Pix"])        

            nums = [] # flood prediction vals
            for i in range(0, geom.GetPointCount()-1):
                pt1 = geom.GetPoint(i)
                pt2 = geom.GetPoint(i+1)
                dist = math.ceil(math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2))
                x0, y0 = pt1[0], pt1[1]
                x1, y1 = pt2[0], pt2[1]
                x, y = np.linspace(x0, x1, dist).astype(int), np.linspace(y0, y1, dist).astype(int)

                for i in range(len(x)):
                    top = max(0, y[i]-dy)
                    bot = min(nrows-1, y[i]+dy)
                    left = max(0, x[i]-dx)
                    right = min(ncols-1, x[i]+dx)
                    nums.extend(flood_arr[top:bot,left:right].flatten())

            currow = row
            binned = np.bincount(nums)
            if len(binned) > 1:
                if binned[flood_road_label] / np.sum(binned) > flood_area_ratio:
                    currow["Flooded"] = "True"

        out_rows.append([currow[k] for k in list(currow.keys())])

    df = pd.DataFrame(out_rows, columns=cols)
    return df


def process_aoi(args, aoi):
    # TODO:
    road_flood_channel = 2
    
    graph_dir = os.path.join(args.graph, aoi)
    if not os.path.exists(graph_dir):
        print(f'graph_dir does not exists for {aoi}')
        return None

    print('graph -> wkt ..')
    df = pkl_dir_to_wkt(
        graph_dir,
        weight_keys=['length'],
        verbose=False
    )

    print('inserting flood attribute ..')
    flood_dir = os.path.join(args.flood, aoi)
    df = insert_flood_pred(
        flood_dir,
        df,
        road_flood_channel,
        args.flood_thresh,
        args.flood_area_ratio,
        args.flood_dilation_ks
    )
    
    return df


def add_empty_rows(args, df, cols):
    image_ids = []
    aois = [d for d in os.listdir(args.flood) if os.path.isdir(os.path.join(args.flood, d))]
    for aoi in aois:
        paths = glob(os.path.join(args.flood, aoi, '*.tif'))
        ids = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        image_ids.extend(ids)
    image_ids.sort()

    empty_rows = []
    for image_id in image_ids:
        # submit without any building prediction
        # this line is removed when concat building and road dataframe
        empty_rows.append([image_id, 'Building', 'POLYGON EMPTY', 'POLYGON EMPTY', 'False', 'Null'])

        if image_id not in list(df.ImageId.unique()):
            # add images where no road was detected
            empty_rows.append([
                image_id, 'Road', 'LINESTRING EMPTY', 'LINESTRING EMPTY', 'False', 'Null'
            ])
    df = df.append(pd.DataFrame(empty_rows, columns=cols))

    return df


def remove_duplicate_linestring_section(df):
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        if row.Object != 'Road':
            continue
        if row.WKT_Pix == 'LINESTRING EMPTY':
            continue

        wkt_geom = loads(row.WKT_Pix)
        points = list(wkt_geom.coords)
        if (points[0] == points[-1]) and (len(points) == 3):
            print('found duplicate LINESTRING section:')
            print(row)
            wkt_geom.coords = wkt_geom.coords[:-1]  # remove the last point
            linestring = dumps(wkt_geom, rounding_precision=0)
            df.at[i, 'WKT_Pix'] = linestring
    return df


def main():
    args = parse_args()
    print(f'flood_thresh={args.flood_thresh}')

    cols = ['ImageId', 'Object', 'WKT_Pix', 'WKT_Geo', 'Flooded', 'length_m']  # 'travel_time_s' will be added later
    df = pd.DataFrame(columns=cols)
    aois = [d for d in os.listdir(args.flood) if os.path.isdir(os.path.join(args.flood, d))]
    for aoi in aois:
        print(f'processing {aoi} AOI')
        ret = process_aoi(args, aoi)
        if ret is not None:
            df = df.append(ret)

    image_ids = []
    for aoi in aois:
        paths = glob(os.path.join(args.flood, aoi, '*.tif'))
        ids = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        image_ids.extend(ids)
    image_ids.sort()

    df = add_empty_rows(args, df, cols)

    df['travel_time_s'] = 'Null'
    df = df.drop(columns='WKT_Geo')

    exp_foundation = os.path.basename(os.path.normpath(args.graph)).replace('exp_', '')
    exp_flood = os.path.basename(os.path.normpath(args.flood)).replace('exp_', '')
    out_dir = f'exp_{exp_foundation}_{exp_flood}'
    if args.val:
        out_dir = os.path.join(args.artifact_dir, '_val/road_submissions', out_dir)
    else:
        out_dir = os.path.join(args.artifact_dir, 'road_submissions', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.val:
        # remove images in the blacklist
       image_id_blacklist = [os.path.splitext(x)[0] for x in pre_image_blacklist]
       df = df[~df.ImageId.isin(image_id_blacklist)]

    # workaround to avoid submission error
    df = remove_duplicate_linestring_section(df)
    df = df.drop_duplicates()

    print(df.head(15))

    out_path = os.path.join(out_dir, 'solution.csv')
    df.to_csv(out_path, index=False)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
