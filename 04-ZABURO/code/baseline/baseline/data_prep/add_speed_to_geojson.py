####################
# Directly from CRESI repo: https://github.com/avanetten/cresi
# Small updates for SpaceNet 8
####################

import os
import sys
import json
import math
import fiona
import shutil
import shapely
import numpy as np
import geopandas as gpd
import random
import skimage
import argparse
import cv2
from osgeo import gdal, ogr, osr
from json import JSONDecodeError

###############################################################################
def speed_func(geojson_row, label_type='sn5', verbose=True):
    '''
    Infer road speed limit based on SpaceNet properties
    # geojson example
    { "type": "Feature", "properties": { "gid": 15806, "road_id": 24791,
            "road_type": 5, "paved": 1, "bridge": 2, "one_way": 2,
            "heading": 0.0, "lane_numbe": 2,
            "ingest_tim": "2017\/09\/24 20:36:06.436+00",
            "edit_date": "2017\/09\/24 20:36:06.436+00",
            "edit_user": "ian_kitchen", "production": "0", "imagery_so": "0",
            "imagery_da": "0", "partialBuilding": 1.0, "partialDec": 0.0 }, 
            "geometry": { "type": "LineString", "coordinates": [ [ -115.305975139809291, 36.179169421086783, 0.0 ], [ -115.305540626738249, 36.179686396492464, 0.0 ], [ -115.305150516462803, 36.180003559318038, 0.0 ], [ -115.304760406187356, 36.18037781145221, 0.0 ], [ -115.304287833577249, 36.180932846396956, 0.0 ], [ -115.304305558679488, 36.18094769983459, 0.0 ] ] } 
    }
    label_type = ['sn5', 'sn3', 'osm']
    if osm_labels, assume an osm label
    Return speed_final_mph, speed_final_mps
    Set < 0 if not found
    '''

    keys = set(geojson_row['properties'].keys())
    # print ("geojson_row:", geojson_row)
    if verbose:
        print("geojson_row", geojson_row)

    if label_type == 'osm':
        # https://wiki.openstreetmap.org/wiki/Key:highway
        # convert road_type_str to int
        conv_dict = {'motorway': 1,
                     'motorway_link': 1,
                     'trunk': 1,
                     'trunk_link': 1,
                     'primary': 2,
                     'primary_link': 2,
                     'bus_guideway': 2,
                     'secondary': 3,
                     'secondary_link': 3,
                     'tertiary': 4,
                     'tertiary_link': 4,
                     'residential': 5,
                     'living_street': 5,
                     'unclassified': 6,
                     'service': 6,
                     'road': 6,
                     'cart_track': 7,
                     'track': 7,
                     }
        skip_set = ('stopline', 'footway', 'bridleway', 'step', 'steps', 
                    'path', 'pedestrian', 'escape', 'cycleway', 'raceway',
                    'bus', 'services', 'escalator', 'sidewalk')


        if ('highway' in geojson_row['properties']):
            road_type_str = geojson_row['properties']['highway']
        elif ('class' in geojson_row['properties']):
            if geojson_row['properties']['class'] == 'highway':
                road_type_str = geojson_row['properties']['type']
            else:
                if verbose:
                    print("  class not highway")
                return -1, -1
        else:
            if verbose:
                print("  not road")
            return -1, -1

        # check type
        if road_type_str in skip_set:
            if verbose:
                print("road_type {} in skip_set".format(road_type_str))
            # print ("  geojson_row['properties']['highway']:", geojson_row['properties']['highway'])
            return -1, -1
        else:
            # print ("  geojson_row['properties']['highway']:", geojson_row['properties']['highway'])
            if verbose:
                print("  road_type_str:", road_type_str)
            road_type = conv_dict[road_type_str.lower()]
            
        # check if tunnel
        if 'tunnel' in keys:
            if geojson_row['properties']['tunnel'] in ['yes']:
                print ("  skipping tunnel!", geojson_row)
                return -1, -1

        num_lanes = 2
        surface = 1
        bridge = 2

    # SpaceNet 3 labels
    elif label_type == 'sn3':
        # sometimes we get a malformed road
        try:
            road_type = int(geojson_row['properties']['road_type'])
            # lane number was incorrectly labeled in initial geojsons
            if 'lane_numbe' in keys:
                num_lanes = int(geojson_row['properties']['lane_numbe'])
            else:
                num_lanes = int(geojson_row['properties']['lane_number'])
            surface = int(geojson_row['properties']['paved'])
            if 'bridge_typ' in keys:
                bridge = int(geojson_row['properties']['bridge_typ'])
            else:
                bridge = int(geojson_row['properties']['bridge_type'])
        except KeyError:
            # assume a one lane unclassified paved road
            road_type = 6
            num_lanes = 1
            surface = 1
            bridge = 2
            print("malformed geojson row:", geojson_row)

    # SpaceNet 5 labels
    elif label_type == 'sn5':
        # e.g., { "type": "Feature", "properties": { "OBJECTID": "43", "bridge": null, "highway": "unclassified", "osm_id": 683685519.000000, "surface": "paved", "lanes": "2" }, "geometry": { "type": "LineString", "coordinates": [ [ 37.633999, 55.626647500000047 ], [ 37.633841700000062, 55.625982300000032 ], [ 37.633794500000079, 55.625240800000029 ], [ 37.633748600000047, 55.625123100000053 ], [ 37.633341800000039, 55.624770700000056 ] ] } },

        highway_conv_dict = {'motorway': 1,
                     'motorway_link': 1,
                     'trunk': 1,
                     'trunk_link': 1,
                     'primary': 2,
                     'primary_link': 2,
                     'bus_guideway': 2,
                     'secondary': 3,
                     'secondary_link': 3,
                     'tertiary': 4,
                     'tertiary_link': 4,
                     'residential': 5,
                     'living_street': 5,
                     'unclassified': 6,
                     'service': 6,
                     'road': 6,
                     'cart_track': 7,
                     'track': 7,
                     }
        # road_type
        road_type_str = geojson_row['properties']['highway']
        road_type = highway_conv_dict[road_type_str.lower()]
        
        # bridge
        bridge_str = geojson_row['properties']['bridge']
        if bridge_str == 'null':
            bridge = 2
        else:
            bridge = 1

        # surface type
        surface_str = geojson_row['properties']['surface']
        if surface_str == 'paved':
            surface = 1
        else:
            surface = 2
        
        # num lanes
        num_lanes = int(float(geojson_row['properties']['lanes']))
        
        if verbose:
            print("road_type:", road_type)
            print("surface:", surface)
            print("num_lanes:", num_lanes)


    # road type (int)
    '''
    1: Motorway
    2: Primary
    3: Secondary
    4: Tertiary
    5: Residential
    6: Unclassified
    7: Cart track
    '''
    #road_type_dict = {
    #    1: 60,
    #    2: 45,
    #    3: 35,
    #    4: 25,
    #    5: 25,
    #    6: 20,
    #    7: 15
    #}

    # https://en.wikipedia.org/wiki/File:Speed_limits_in_Ohio.svg
    # https://wiki.openstreetmap.org/wiki/OSM_tags_for_routing/Maxspeed#United_States_of_America    
    #   Use Oregon:
    #       State	Motorway	Trunk	Primary	Secondary	Tertiary	Unclassified	Residential	Living street	Service
    #       Oregon	55 mph	55 mph	55 mph	35 mph	30 mph		         25 mph		              15 mph
    # feed in [road_type][num_lanes]
    nested_speed_dict = {
        1: {1: 55, 2: 55, 3: 65, 4: 65, 5: 65, 6: 65, 7: 65, 8: 65, 9: 65, 10: 65, 11: 65, 12: 65},
        2: {1: 45, 2: 45, 3: 55, 4: 55, 5: 55, 6: 55, 7: 55, 8: 55, 9: 55, 10: 55, 11: 55, 12: 55},
        3: {1: 35, 2: 35, 3: 45, 4: 45, 5: 45, 6: 45, 7: 45, 8: 45, 9: 45, 10: 45, 11: 45, 12: 45},
        4: {1: 30, 2: 30, 3: 35, 4: 35, 5: 35, 6: 35, 7: 35, 8: 35, 9: 35, 10: 35, 11: 35, 12: 35},
        5: {1: 25, 2: 25, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30, 8: 30, 9: 30, 10: 30, 11: 30, 12: 30},
        6: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20},
        7: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20}
    }

    # multiply speed by this factor based on surface
    # 1 = paved, 2 = unpaved
    road_surface_dict = {
        1: 1,
        2: 0.75
    }

    # multiply speed by this factor for bridges
    # 1 = bridge, 2 = not bridge
    bridge_dict = {
        1: 1,  # 0.8,
        2: 1}

#    # V0, Feb 22 2019 and prior
#    # feed in [road_type][num_lanes]
#    nested_speed_dict = {
#        1: {1: 45, 2: 50, 3: 55, 4: 65, 5: 65, 6: 65, 7: 65, 8: 65, 9: 65, 10: 65, 11: 65, 12: 65},
#        2: {1: 35, 2: 40, 3: 45, 4: 45, 5: 45, 6: 45, 7: 45, 8: 45, 9: 45, 10: 45, 11: 45, 12: 45},
#        3: {1: 30, 2: 30, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30, 8: 30, 9: 30, 10: 30, 11: 30, 12: 30},
#        4: {1: 25, 2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25, 12: 25},
#        5: {1: 25, 2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25, 12: 25},
#        6: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20},
#        7: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20}
#    }
#    # multiply speed by this factor based on surface
#    road_surface_dict = {
#        1: 1,
#        2: 0.75
#    }
#    # multiply speed by this factor for bridges
#    bridge_dict = {
#        1: 1, #0.8,
#        2: 1}

    # default speed in miles per hour
    speed_init_mph = nested_speed_dict[road_type][num_lanes]
    # reduce speed for unpaved or bridge
    speed_final_mph = speed_init_mph * road_surface_dict[surface] \
        * bridge_dict[bridge]
    # get speed in meters per second
    speed_final_mps = 0.44704 * speed_final_mph

    if verbose:
        print("speed_mph:", speed_final_mph)

    return speed_final_mph, speed_final_mps

###############################################################################
def add_speed_to_geojson(geojson_path_in, geojson_path_out,
                         label_type='sn5',
                         speed_key_mph='inferred_speed_mph',
                         speed_key_mps='inferred_speed_mps',
                         verbose=True):
    '''Update geojson data to add inferred speed information'''
    
    """ added a catch to only calcualte speed on road features. will ignore features when attribute 'building'=='yes' """

    speed_mph_set = set()
    speed_mph_arr = []
    bad_row_idxs = []
    with open(geojson_path_in, 'r+') as f:
        try:
            geojson_data = json.load(f)
        except JSONDecodeError:
            # assume empty array, copy
            shutil.copy(geojson_path_in, geojson_path_out)
            return [], set()

        init_len = len(geojson_data['features'])
        for i, geojson_row in enumerate(geojson_data['features']):
            if verbose and (i % 100) == 0:
                print("\n", i, "/", init_len, "geojson_row:", geojson_row)

            # optional: also update "ingest_tim" tag
            if 'ingest_tim' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['ingest_tim']
                del geojson_row['properties']['ingest_tim']
                geojson_row['properties']['ingest_time'] = x_tmp

            # optional: also update "bridge_typ" tag
            if 'bridge_typ' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['bridge_typ']
                del geojson_row['properties']['bridge_typ']
                geojson_row['properties']['bridge_type'] = x_tmp

            # optional: also update "lane_numbe" tag
            if 'lane_numbe' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['lane_numbe']
                del geojson_row['properties']['lane_numbe']
                geojson_row['properties']['lane_number'] = x_tmp

            if geojson_row["properties"]["building"] != "yes": # only calculate speed for roads. not buildings
                # infer route speed limit
                speed_mph, speed_mps = speed_func(geojson_row, 
                                                  label_type=label_type,
                                                  verbose=verbose)
                if verbose:
                    print("  speed_mph, speed_mps:", speed_mph, speed_mps)
                if speed_mph >= 0:
                    # update properties
                    geojson_row['properties'][speed_key_mph] = speed_mph
                    geojson_row['properties'][speed_key_mps] = speed_mps
                    speed_mph_set.add(speed_mph)
                    speed_mph_arr.append(speed_mph)
                else:
                    if verbose:
                        print("geojson_row:", geojson_row)
                    bad_row_idxs.append(i)
    
    # remove bad idxs
    if len(bad_row_idxs) > 0:
        # geojson_data['features'] = [e+'\n' for i, e in
        geojson_data['features'] = [e+'\n' for i, e in
                enumerate(geojson_data['features']) if i not in bad_row_idxs]
        final_len = len(geojson_data['features'])
        # print("  bad_row_idxs:", bad_row_idxs)
        if 2 > 1: #verbose:
            print("  init_len:", init_len, "final_len:", final_len)

    # save file
    with open(geojson_path_out, 'w') as f:
        f.write(json.dumps(geojson_data, indent=2))
        if verbose:
            print("geojson_path_out:", geojson_path_out)
        # f.write(json.dumps(geojson_data, indent=1))

#    # older version that doesn't print correctly
#    geojson_data = fiona.open(geojson_path, 'r')
#    out = []
#    for i,geojson_row in enumerate(geojson_data):
#        if verbose:
#            print ("\ngeojson_row:", geojson_row)
#        # infer route speed limit
#        speed_mph, speed_mps = speed_func(geojson_row)
#        if verbose:
#            print ("  speed_mph, speed_mps:", speed_mph, speed_mps)
#        # update properties
#        geojson_row['properties']['speed_mph'] = speed_mph
#        geojson_row['properties']['speed_m/s'] = speed_mps
#        #out.append(geojson_row)
#    # save file
#    with open(geojson_path_out, 'w') as f:
#        json.dump(out, f, ensure_ascii=False)

    return speed_mph_arr, speed_mph_set

###############################################################################
def update_geojson_dir_speed(geojson_dir_in, geojson_dir_out,
                             label_type='sn5', suffix='_speed', nmax=1000000,
                             verbose=True, super_verbose=False):
    '''Update geojson data to add inferred speed information for entire
    directory'''

    os.makedirs(geojson_dir_out, exist_ok=True)
    speed_mph_set = set()
    speed_mph_arr = []

    json_files = np.sort([j for j in os.listdir(geojson_dir_in)
                          if j.endswith('.geojson')])
    if super_verbose:
        print("json_files:", json_files)

    for i, json_file in enumerate(json_files):
        if i >= nmax:
            break
        if (i % 100) == 0 and verbose:  # verbose:
            print(i, "/", len(json_files), json_file)
        geojson_path_in = os.path.join(geojson_dir_in, json_file)
        root, ext = json_file.split('.')
        geojson_path_out = os.path.join(geojson_dir_out,
                                        root + suffix + '.' + ext)
        sarr, sset = add_speed_to_geojson(geojson_path_in, geojson_path_out,
                                          label_type=label_type,
                                          verbose=verbose)
        speed_mph_arr.extend(sarr)
        speed_mph_set = speed_mph_set.union(sset)

    if verbose:
        print("speed_mph_set:", sorted(list(speed_mph_set)))
    unique, counts = np.unique(speed_mph_arr, return_counts=True)
    if verbose:
        print("speed_mph counts:")
    d = dict(zip(unique, counts))
    if verbose:
        for k, v in d.items():
            print(" ", k, v)

    return

def update_geojson_file_speed(geojson_file_in, geojson_file_out,
                              label_type='sn5', suffix='_speed', nmax=1000000,
                             verbose=True, super_verbose=False):
    """Update geojson data to add inferred speed information for a single file"""
    speed_mph_set = set()
    speed_mph_arr = []
    
    if super_verbose:
        print("json_file:", geojson_file_in)

    sarr, sset = add_speed_to_geojson(geojson_file_in, geojson_file_out,
                                      label_type=label_type,
                                      verbose=verbose)
    speed_mph_arr.extend(sarr)
    speed_mph_set = speed_mph_set.union(sset)

    if verbose:
        print("speed_mph_set:", sorted(list(speed_mph_set)))
    unique, counts = np.unique(speed_mph_arr, return_counts=True)
    if verbose:
        print("speed_mph counts:")
    d = dict(zip(unique, counts))
    if verbose:
        for k, v in d.items():
            print(" ", k, v)

    return