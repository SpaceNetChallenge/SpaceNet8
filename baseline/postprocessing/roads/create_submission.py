"""
From: https://github.com/avanetten/cresi

with small modifications

"""

import os
import json
import time
import argparse
import math
import pandas as pd
import networkx as nx
import numpy as np
import glob

from osgeo import osr
from osgeo import ogr
from osgeo import gdal

###############################################################################
def write_road_submission_shapefile(df, out_shapefile):
    df = df.reset_index()  # make sure indexes pair with number of rows

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    # Create the output shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(out_shapefile)
    out_layer = ds.CreateLayer(out_shapefile[:-4], srs, ogr.wkbLineString)
    
    fieldnames = ['ImageId', 'Object', 'Flooded', 'WKT_Pix', 'WKT_Geo', 'length_m', 'travel_time_s']
    
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
    out_layer.CreateField(ogr.FieldDefn('travel_t_s', ogr.OFTReal))

    # Create the feature and set values
    featureDefn = out_layer.GetLayerDefn()
    
    for index, row in df.iterrows():
        
        outFeature = ogr.Feature(featureDefn)
        for j in fieldnames:
            if j == "travel_time_s":
                outFeature.SetField('travel_t_s', row[j])
            else:
                outFeature.SetField(j, row[j])
        
        geom = ogr.CreateGeometryFromWkt(row["WKT_Geo"])
        outFeature.SetGeometry(geom)
        out_layer.CreateFeature(outFeature)
        outFeature = None
    ds = None

def pkl_dir_to_wkt(pkl_dir, output_csv_path='',
                   weight_keys=['length', 'travel_time_s'],
                   verbose=False):
    """
    Create submission wkt from directory full of graph pickles
    """
    wkt_list = []

    pkl_list = sorted([z for z in os.listdir(pkl_dir) if z.endswith('.gpickle')])
    for i, pkl_name in enumerate(pkl_list):
        print(pkl_name)
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
            wkt_item_root = [name_root, 'LINESTRING EMPTY']
            if len(weight_keys) > 0:
                weights = [0 for w in weight_keys]
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

            wkt_item_root = [name_root, 'Road', 'False', geom_pix_wkt, geom_geo_wkt]
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
        cols = ['ImageId', 'Object', 'Flooded', 'WKT_Pix', 'WKT_Geo'] + weight_keys
    else:
        cols = ['ImageId', 'Object', 'Flooded', 'WKT_Pix', 'WKT_Geo']

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
    print("cols:", cols)

    df = pd.DataFrame(wkt_list, columns=cols)
    print("df:", df)
    # save
    #if len(output_csv_path) > 0:
    #    df.to_csv(output_csv_path, index=False)
    return df

def insert_flood_pred(flood_pred_dir, df, output_csv_path, output_shapefile_path):
    
    #flood_preds = glob.glob(os.path.join(flood_pred_dir, "*floodpred.tif"))
    
    dy=2
    dx=2
    cols = ['ImageId', 'Object', 'Flooded', 'WKT_Pix', 'WKT_Geo', 'length_m', 'travel_time_s']
    out_rows = []
    for index, row in df.iterrows():
        imageid = row["ImageId"]
        
        flood_pred_filename = os.path.join(flood_pred_dir, f"{imageid}_floodpred.tif")
        assert(os.path.exists(flood_pred_filename)), "flood prediction file for this linestring doesn't exist"
        ds = gdal.Open(flood_pred_filename)
        nrows = ds.RasterYSize
        ncols = ds.RasterXSize
        flood_arr = ds.ReadAsArray()

        if row["Object"] != "LINESTRING EMPTY":
            geom = ogr.CreateGeometryFromWkt(row["WKT_Pix"])        

            nums = [] # flood prediction vals
            for i in range(0, geom.GetPointCount()-1):
                pt1 = geom.GetPoint(i)
                pt2 = geom.GetPoint(i+1)
                dist = math.ceil(math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2))
                x0, y0 = pt1[0], pt1[1]
                x1, y1 = pt2[0], pt2[1]
                x, y = np.linspace(x0, x1, dist).astype(np.int), np.linspace(y0, y1, dist).astype(np.int)

                for i in range(len(x)):
                    top = max(0, y[i]-dy)
                    bot = min(nrows-1, y[i]+dy)
                    left = max(0, x[i]-dx)
                    right = min(ncols-1, x[i]+dx)
                    nums.extend(flood_arr[top:bot,left:right].flatten())

            currow = row
            maxval = np.argmax(np.bincount(nums))
            if maxval == 4:
                currow["Flooded"] = "True"

        out_rows.append([currow[k] for k in list(currow.keys())])


    df = pd.DataFrame(out_rows, columns=cols)
    #print("df:", df)
    # save
    if len(output_csv_path) > 0:
        df.to_csv(output_csv_path, index=False)
    
    write_road_submission_shapefile(df, output_shapefile_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flood_pred",
                        type=str,
                        required=True)
    parser.add_argument("--graph_speed_dir",
                        type=str,
                        required=True)
    parser.add_argument("--output_csv_path",
                        type=str,
                        required=True)
    parser.add_argument("--output_shapefile_path",
                        type=str,
                        required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    flood_pred = args.flood_pred
    graph_speed_dir = args.graph_speed_dir
    output_csv_path = args.output_csv_path
    output_shapefile_path = args.output_shapefile_path

    # Final
    t0 = time.time()
    weight_keys = ['length', 'travel_time_s']
    verbose = False #True

    df = pkl_dir_to_wkt(graph_speed_dir,
                        output_csv_path=output_csv_path,
                        weight_keys=weight_keys, verbose=verbose)

    insert_flood_pred(flood_pred, df, output_csv_path, output_shapefile_path)
    tf = time.time()
    print("Submission file:", output_csv_path)
    print("Time to create submission:", tf-t0, "seconds")
