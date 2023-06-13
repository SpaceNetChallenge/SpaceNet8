import json
import copy
import glob
import os
import argparse

from shapely.geometry import mapping, shape
import fiona
from fiona.crs import from_epsg

from add_speed_to_geojson import *

# common schemas for buildings and roads
BUILDING_SCHEMA = {
    'geometry': 'Polygon',
    'properties': {'id': 'str',
                   'id2': 'str',
                   'building': 'str',
                   'flooded': 'str',
                   'ghosted': 'str',
                   'highway': 'str',
                   'lanes': 'str',
                   'surface': 'str',
                   'oneway': 'str',
                   'lanes:forward': 'str',
                   'bridge': 'str',
                   'tunnel': 'str',
                   'layer': 'str',
                   'mask_id':'int',
                   'speed_mph':'float'}
}

ROAD_SCHEMA = {
    'geometry': 'LineString',
    'properties': {'id': 'str',
                   'id2': 'str',
                   'building': 'str',
                   'flooded': 'str',
                   'ghosted': 'str',
                   'highway': 'str',
                   'lanes': 'str',
                   'surface': 'str',
                   'oneway': 'str',
                   'lanes:forward': 'str',
                   'bridge': 'str',
                   'tunnel': 'str',
                   'layer': 'str',
                   'mask_id':'int',
                   'speed_mph':'float'}
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",
                        required=True,
                        type=str,
                        help="path to the directory holding the AOIs directories")
    parser.add_argument("--aoi_dirs",
                        required=True,
                        nargs="+",
                        type=str,
                        help="directory names of the AOIs (e.g. Germany_Training_Public, Louisiana-East_Training_Public)")
    args = parser.parse_args()
    return args

def match_im_label(annotations, pre_images, post_images):
    out_pre = []
    out_anno = []
    out_post = []
    for i in annotations:
        tileid = os.path.basename(i).split(".")[0]
        pre_im = [j for j in pre_images if f"_{tileid}.tif" in j][0]
        post_im = [j for j in post_images if f"_{tileid}.tif" in j][0]
        
        out_anno.append(i)
        out_pre.append(pre_im)
        out_post.append(post_im)
    return out_anno, out_pre, out_post

# getters for road, building, flood features
def get_building_features(geojson_dict):
    out_features = []
    for i in geojson_dict["features"]:
        if "building" in i["properties"] and i["properties"]["building"] is not None:
            out_features.append(i)
    return out_features

def get_road_features(geojson_dict):
    # road records are: 'building'=None or 'building' not in the record's attributes at all. 
    out_features = []
    for i in geojson_dict["features"]:
        if "highway" in i["properties"] and i["properties"]["highway"] is not None:
            out_features.append(i)
    return out_features

def merge_to_common_properties_schema(geojson_features):
    """
    normalize data

    There are differences in the geojson schemas depending on AOI. This function
    is used to create a common schema across all AOIs. When a property does not exist
    in the schema, it is set to None.
    """
    properties_template = {'id': None,
                           '@id': None,
                           'building': None,
                           'flooded': None,
                           'ghosted': None,
                           'highway': None,
                           'lanes': None,
                           'surface': None,
                           'oneway': None,
                           'lanes:forward': None,
                           'bridge': None,
                           'tunnel': None,
                           'layer': None}
    out_features = []
    for i in geojson_features:
        props = copy.copy(properties_template)
        for j in i['properties']:
            props[j] = i['properties'][j]
        modified_record = {'type': 'Feature',
                           'properties': props,
                           'geometry': i['geometry']}
        
        out_features.append(modified_record)
    return out_features
    
def remove_bad_features(geojson_features):
    """
    cleaning and removing bad labels.
    
    Remove features where "building" is Null and "highway" is Null.
    Remove features where "highway" is not Null and "lanes" is Null
    """
    out_features = []
    bad_features = []
    for i in geojson_features:
        if i["properties"]["building"] is None and i["properties"]["highway"] is None:
            bad_features.append(i["properties"])
        elif i["properties"]["highway"] is not None and i["properties"]["lanes"] is None:
            bad_features.append(i["properties"])
        else:
            out_features.append({'type': 'Feature',
                               'properties': i["properties"],
                               'geometry': i['geometry']})
    #print(f"removed {len(bad_features)} bad features: ")
    #for i in range(len(bad_features)):
    #    print(i, bad_features[i])
    return out_features
    
    
def geojson_to_shapefile(geojson_features, output_shapefile_filename, schema, crs):
    with fiona.open(output_shapefile_filename, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as c:
        for i in geojson_features:
            s = shape(i['geometry'])
            props = i['properties']
            if props['flooded'] in ['yes', 1, True]:
                mask_id = 2
            else:
                mask_id = 1
                
            if props['highway'] is not None:
                speed = props['inferred_speed_mph']
            else:
                speed = None
            c.write({
                'geometry': mapping(s),
                'properties': {'id': props['id'],
                               'id2': props['@id'],
                               'building': props['building'],
                               'flooded': props['flooded'],
                               'ghosted': props['ghosted'],
                               'highway': props['highway'],
                               'lanes': props['lanes'],
                               'surface': props['surface'],
                               'oneway': props['oneway'],
                               'lanes:forward': props['lanes:forward'],
                               'bridge': props['bridge'],
                               'tunnel': props['tunnel'],
                               'layer': props['layer'],
                               'mask_id':mask_id,
                               'speed_mph':speed},
            })

def save_geojson_features(geojson_features, outfile):
    outdict = {"type":"FeatureCollection",
               "features":geojson_features}
    with open(outfile, 'w') as outjson:
        json.dump(outdict, outjson)
            
def create_buildings_roads_shp(infile):
    f = open(infile)
    data = json.load(f)
    f.close()

    crs = from_epsg(4326)

    input_features = get_road_features(data)
    merged_features = merge_to_common_properties_schema(input_features)
    cleaned_features = remove_bad_features(merged_features)
    save_geojson_features(cleaned_features, f"{infile[:-8]}_roads_cleaned.geojson")
    
    outfilename = f"{infile[:-8]}_roads.shp"
    geojson_to_shapefile(cleaned_features, outfilename, ROAD_SCHEMA, crs)
    
    input_features = get_building_features(data)
    merged_features = merge_to_common_properties_schema(input_features)
    cleaned_features = remove_bad_features(merged_features)
    save_geojson_features(cleaned_features, f"{infile[:-8]}_buildings_cleaned.geojson")
    outfilename = f"{infile[:-8]}_buildings.shp"
    geojson_to_shapefile(cleaned_features, outfilename, BUILDING_SCHEMA, crs)

def clean(in_geojson_filename, out_dir):
    f = open(in_geojson_filename)
    data = json.load(f)
    f.close()

    input_features = get_road_features(data)
    merged_features = merge_to_common_properties_schema(input_features)
    cleaned_features = remove_bad_features(merged_features)
    save_geojson_features(cleaned_features, f"{os.path.join(out_dir, 'roads_cleaned_'+os.path.basename(in_geojson_filename).split('.')[0])}.geojson")
    
    input_features = get_building_features(data)
    merged_features = merge_to_common_properties_schema(input_features)
    cleaned_features = remove_bad_features(merged_features)
    save_geojson_features(cleaned_features, f"{os.path.join(out_dir, 'buildings_cleaned_'+os.path.basename(in_geojson_filename).split('.')[0])}.geojson")
    
def add_road_speed(in_geojson_filename, out_geojson_filename):
    update_geojson_file_speed(in_geojson_filename, out_geojson_filename,
                              label_type='sn5', suffix='_speed', nmax=1000000,
                             verbose=False, super_verbose=False)
    
def write_shapefile(in_geojson_filename, out_shapefile_filename, schema):
    f = open(in_geojson_filename)
    data = json.load(f)
    f.close()
    
    crs = from_epsg(4326)
    
    features = data['features']
    geojson_to_shapefile(features, out_shapefile_filename, schema, crs)

def do_data_prep(in_geojson_filename, out_dir):
    """ 
    1. clean both roads and buildings - catches geometry problems, makes a single commom schema.
        output: _roads_cleaned.geojson, _buildings_cleaned.geojson
    2. add road speed to roads
        output: _roads_speed.geojson
    3. write shapefiles for roads and buildings
        output: _roads_speed.shp, _buildings.shp
    """
    print(in_geojson_filename)
    
    clean(in_geojson_filename, out_dir) # clean both roads and buildings geojson

    cleaned_roads_geojson = f"{os.path.join(out_dir, 'roads_cleaned_'+os.path.basename(in_geojson_filename).split('.')[0])}.geojson"
    cleaned_buildings_geojson = f"{os.path.join(out_dir, 'buildings_cleaned_'+os.path.basename(in_geojson_filename).split('.')[0])}.geojson"
    
    speed_roads_geojson = f"{os.path.join(out_dir, 'roads_speed_'+os.path.basename(in_geojson_filename).split('.')[0])}.geojson"
    add_road_speed(cleaned_roads_geojson, speed_roads_geojson) # add speed to roads geojson
    
    out_roads_shapefile = f"{os.path.join(out_dir, 'roads_'+os.path.basename(in_geojson_filename).split('.')[0])}.shp"
    write_shapefile(speed_roads_geojson, out_roads_shapefile, ROAD_SCHEMA) # write roads
    
    out_buildings_shapefile = f"{os.path.join(out_dir, 'buildings_'+os.path.basename(in_geojson_filename).split('.')[0])}.shp"
    write_shapefile(cleaned_buildings_geojson, out_buildings_shapefile, BUILDING_SCHEMA) # write buildings

if __name__ == "__main__":

    args = parse_args()
    root_dir = args.root_dir
    aoi_dirs = args.aoi_dirs

    geojsons = []
    pre_images = []
    post_images = []
    for i in aoi_dirs:
        assert(os.path.exists(os.path.join(root_dir, i))), "error: aoi doesn't exist"
        
        anno = glob.glob(os.path.join(root_dir, i, "annotations", "*.geojson"))
        pre = glob.glob(os.path.join(root_dir, i, "PRE-event", "*.tif"))
        post = glob.glob(os.path.join(root_dir, i, "POST-event", "*.tif"))

        print("number of annotations: ", len(anno))
        print("number of pre: ", len(pre))
        print("number of post: ", len(post))

        annot, preims, postims = match_im_label(anno, pre, post)

        print("number of annotations: ", len(annot))
        print("number of pre: ", len(preims))
        print("number of post: ", len(postims))
        
        geojsons.extend(annot)
        pre_images.extend(preims)
        post_images.extend(postims)

    for i in geojsons:
        out_dir = os.path.join(os.path.dirname(i), "prepped_cleaned")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            os.chmod(out_dir, 0o777)
        do_data_prep(i, out_dir)