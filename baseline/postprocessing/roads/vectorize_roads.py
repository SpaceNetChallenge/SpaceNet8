import argparse

from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, medial_axis
from skimage.morphology import erosion, dilation, opening, closing, disk
import numpy as np
from scipy import ndimage as ndi
from matplotlib.pylab import plt
import os
import pandas as pd
from itertools import tee
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict #, defaultdict
import json
import time
import random
import argparse
import networkx as nx
import logging
from multiprocessing.pool import Pool
import skimage
import skimage.draw
import scipy.spatial
import skimage.io 
import cv2
import glob

from osgeo import gdal
from osgeo import osr
from osgeo import ogr

from utils import sknw, sknw_int64

linestring = "LINESTRING {}"

def parse_args():
    parser = argparse.ArgumentParser(description="vectorize road masks")

    parser.add_argument('--image_filename',
                        help='the raw image filename',
                        required=False,
                        type=str)
    parser.add_argument('--im_dir',
                        help='a directory full of predictions masks (.tif)',
                        required=False,
                        default="",
                        type=str)
    parser.add_argument('--out_dir',
                        help='the output directory to write output to',
                        required=False,
                        default="",
                        type=str)
    parser.add_argument('--out_shp',
                        help='output filename for the shapefile',
                       required=False)
    parser.add_argument('--out_skeleton_filename',
                        help="output filename for the skeleton raster. it is optional.",
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('--out_graph_filename',
                        help="the output graph filename (.pkl). optional",
                       required=False,
                        type=str)
    parser.add_argument('--out_csv',
                        help='output filename for a csv containing the linestrings. optional',
                       default=None,
                       required=False)
    parser.add_argument('--write_shps',
                        help='only valid if out_dir and im_dir are specified.',
                       action='store_true')
    parser.add_argument('--write_csvs',
                        help='only valid if out_dir and im_dir are specified.',
                       action='store_true')
    parser.add_argument('--write_graphs',
                        help='only valid if out_dir and im_dir are specified.',
                       action='store_true')
    parser.add_argument('--write_skeletons',
                        help='only valid if out_dir and im_dir are specified.',
                       action='store_true')
    args = parser.parse_args()
    return args
    

###############################################################################
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


###############################################################################
def remove_sequential_duplicates(seq):
    # todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res


###############################################################################
def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


###############################################################################
def flatten(l):
    return [item for sublist in l for item in sublist]


###############################################################################
def get_angle(p0, p1=np.array([0, 0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1) 
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

#####################################################################################
def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines

###############################################################################
def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

###############################################################################
def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


###############################################################################
def add_small_segments(G, terminal_points, terminal_lines, 
                       dist1=24, dist2=80,
                       angle1=30, angle2=150, 
                       verbose=False):
    '''Connect small, missing segments
    terminal points are the end of edges.  This function tries to pair small
    gaps in roads.  It will not try to connect a missed T-junction, as the 
    crossroad will not have a terminal point'''
    
    print("Running add_small_segments()")
    try:
        node = G.node
    except:
        node = G.nodes
    # if verbose:
    #   print("node:", node)

    term = [node[t]['o'] for t in terminal_points]
    # print("term:", term)
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < dist1))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > dist1) & (dists < dist2))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > dist1:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if (-1*angle1 < angle < angle1) or (angle < -1*angle2) or (angle > angle2):
            good_pairs.append((s, e))

    if verbose:
        print("  good_pairs:", good_pairs)
        
    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.nodes[s]['o'], G.nodes[e]['o']]
        # print("s_d", s_d)
        # print("type s_d", type(s_d))
        # print("s_d - e_d", s_d - e_d)
        # return
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    good_coords = []
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.nodes[s]['o'].astype(np.int32), G.nodes[e]['o'].astype(np.int32)
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
            good_coords.append( (tuple(s_d), tuple(e_d)) )
    return wkt, good_pairs, good_coords

def G_to_wkt(G, add_small=True, connect_crossroads=True,
             img_copy=None, debug=False, verbose=False, super_verbose=False):
    """Transform G to wkt"""

    # print("G:", G)
    if G == [linestring.format("EMPTY")] or type(G) == str:
        return [linestring.format("EMPTY")]

    node_lines = graph2lines(G)
    # if verbose:
    #    print("node_lines:", node_lines)

    if not node_lines:
        return [linestring.format("EMPTY")]
    try:
        node = G.node
    except:
        node = G.nodes
    # print("node:", node)
    deg = dict(G.degree())
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]

    # refine wkt
    if verbose:
        print("Refine wkt...")
    terminal_lines = {}
    vertices = []
    for i, w in enumerate(node_lines):
        if ((i % 10000) == 0) and (i > 0) and verbose:
            print("  ", i, "/", len(node_lines))
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]['o'], node[e]['o']
                # print("s_coord:", s_coord, "e_coord:", e_coord)
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        # print("segments:", segments)
        # return
    
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        small_segs, good_pairs, good_coords = add_small_segments(
            G, terminal_points, terminal_lines, verbose=verbose)
        print("small_segs", small_segs)
        wkt.extend(small_segs)

    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return [linestring.format("EMPTY")]

    #return cross_segs
    return wkt

def write_shapefile_from_wkt_list(out_shp, reference_image_filename, wkt_list):
    #wkt_list has linestrings with coordinates in image space. i.e. (row, col). convert these to geo coords
    refds = gdal.Open(reference_image_filename)
    geotran = refds.GetGeoTransform()
    xres = geotran[1]
    yres = geotran[5]
    xmin = geotran[0]
    ymax = geotran[3]
    ds_projection_ref = refds.GetProjectionRef()
    
    proj = osr.SpatialReference(wkt=refds.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)
    refds = None
    
    # set up the output shapefile 
    out_driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the data source
    out_datasource = out_driver.CreateDataSource(out_shp)
    # create the layer
    in_srs = osr.SpatialReference()
    print(epsg)
    in_srs.ImportFromEPSG(int(epsg))
    out_layer = out_datasource.CreateLayer(out_shp[:-4], in_srs, ogr.wkbLineString)

    if wkt_list != [linestring.format("EMPTY")]:
        for l in wkt_list:
            coords_string = l[l.index("("):]
            coords_list = coords_string.split(",")
            points = []
            for i in coords_list:
                p = i[1:].replace(")","").split(" ")
                for j in range(len(p)):
                    p[j] = float(p[j])
                points.append(p)
            outstring = "LINESTRING ("
            for p in range(len(points)):
                x1, y1 = points[p][0], points[p][1]
                x1 = xmin + (x1*xres)
                y1 = ymax + (y1*yres)
                if p == 0:
                    outstring += str(x1) + " " + str(y1)
                else:
                    outstring += ", " + str(x1) + " " + str(y1)
            outstring += ")"
            geom_wkt = outstring
                
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            line_geom = ogr.CreateGeometryFromWkt(geom_wkt)
            # Set the feature geometry using the polygon
            out_feature.SetGeometry(line_geom)
            # Create the feature in the layer (shapefile)
            out_layer.CreateFeature(out_feature)
            out_feature = None
    

def build_wkt_linestrings(G,
                          outfile,
                          outshp,
                          reference_image_filename,
                          add_small=True,
                          verbose=True,
                          super_verbose=True):# now build wkt_list (single-threaded)
    all_data = []
    t1 = time.time()

    #G = nx.read_gpickle(gpickle) # this is to load from a saved .pkl graph file
    wkt_list = G_to_wkt(G, add_small=add_small, 
                        verbose=verbose, super_verbose=super_verbose)
    
    write_shapefile_from_wkt_list(outshp, reference_image_filename, wkt_list)
    # add to all_data
    for v in wkt_list:
        #all_data.append((gpickle, v))
        #orig_fname = reference_image_filename.replace("roadspeedpred.tif","preimg.tif")
        orig_fname = reference_image_filename
        all_data.append((orig_fname, v))
    t2 = time.time()
    print("Time to build graph: {} seconds".format(t2-t1))
    
    # save to csv
    df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    if outfile != None:
        df.to_csv(outfile, index=False)
    return df

def remove_small_terminal(G, weight='weight', min_weight_val=30, 
                          pix_extent=1300, edge_buffer=4, verbose=False):
    '''Remove small terminals, if a node in the terminal is within edge_buffer
    of the the graph edge, keep it'''
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    if verbose:
        print("remove_small_terminal() - N terminal_points:", len(terminal_points))
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
            
        # check if at edge
        sx, sy = G.nodes[s]['o']
        ex, ey = G.nodes[e]['o']
        edge_point = False
        for ptmp in [sx, sy, ex, ey]:
            if (ptmp < (0 + edge_buffer)) or (ptmp > (pix_extent - edge_buffer)):
                if verbose:
                    print("ptmp:", ptmp)
                    print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                    print("(ptmp > (pix_extent - edge_buffer):", (ptmp > (pix_extent - edge_buffer)))
                    print("ptmp < (0 + edge_buffer):", (ptmp < (0 + edge_buffer)))
                edge_point = True
            else:
                continue
        # don't remove edges near the edge of the image
        if edge_point:
            if verbose:
                print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                print("edge_point:", sx, sy, ex, ey, "continue")
            continue

        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if verbose:
                print("val.get(weight, 0):", val.get(weight, 0) )
            if s in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(s)
            if e in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(e)
    return

def img_to_ske_G(img_refine, 
                 ske, 
                 min_spur_length_pix,
                 out_gpickle='',
                 verbose=True):
    #out_gpickle = '' # the graph will not be saved as a .pkl

    
    # create graph
    if verbose:
        print("Execute sknw...")
    # if the file is too large, use sknw_int64 to accomodate high numbers for coordinates
    if np.max(ske.shape) > 32767:
        G = sknw_int64.build_sknw(ske, multi=True)
    else:
        G = sknw.build_sknw(ske, multi=True)

   # print a random node and edge
    if verbose:
        node_tmp = list(G.nodes())[-1]
        print(node_tmp, "random node props:", G.nodes[node_tmp])
        # print an edge
        edge_tmp = list(G.edges())[-1]
        #print("random edge props for edge:", edge_tmp, " = ",
        #      G.edges[edge_tmp[0], edge_tmp[1], 0]) #G.edge[edge_tmp[0]][edge_tmp[1]])

    # iteratively clean out small terminals
    for itmp in range(8):
        ntmp0 = len(G.nodes())
        if verbose:
            print("Clean out small terminals - round", itmp)
            print("Clean out small terminals - round", itmp, "num nodes:", ntmp0)
        # sknw attaches a 'weight' property that is the length in pixels
        pix_extent = np.max(ske.shape)
        remove_small_terminal(G, weight='weight',
                              min_weight_val=min_spur_length_pix,
                              pix_extent=pix_extent)
        # kill the loop if we stopped removing nodes
        ntmp1 = len(G.nodes())
        if ntmp0 == ntmp1:
            break
        else:
            continue

    if verbose:
        print("len G.nodes():", len(G.nodes()))
        print("len G.edges():", len(G.edges()))
    if len(G.edges()) == 0:
        return [linestring.format("EMPTY"), [], []]

    # print a random node and edge
    if verbose:
        node_tmp = list(G.nodes())[-1]
        print(node_tmp, "random node props:", G.nodes[node_tmp])
        # print an edge
        edge_tmp = list(G.edges())[-1]
        print("random edge props for edge:", edge_tmp, " = ",
              G.edges[edge_tmp[0], edge_tmp[1], 0]) #G.edge[edge_tmp[0]][edge_tmp[1]])
        # node_tmp = list(G.nodes())[np.random.randint(len(G.nodes()))]
        # print(node_tmp, "G.node props:", G.nodes[node_tmp])
        # edge_tmp = list(G.edges())[np.random.randint(len(G.edges()))]
        # print(edge_tmp, "G.edge props:", G.edges(edge_tmp))
        # print(edge_tmp, "G.edge props:", G.edges[edge_tmp[0]][edge_tmp[1]])

    # # let's not clean out subgraphs yet, since we still need to run
    # # add_small_segments() and terminal_points_to_crossroads()
    # if verbose:
    #     print("Clean out short subgraphs")
    #     try:
    #         sub_graphs = list(nx.connected_component_subgraphs(G))
    #     except:
    #         sub_graphs = list(nx.conncted_components(G))
    #     # print("sub_graphs:", sub_graphs)
    # # sknw attaches a 'weight' property that is the length in pixels
    # t01 = time.time()
    # G = clean_sub_graphs(G, min_length=min_subgraph_length_pix,
    #                  max_nodes_to_skip=100,
    #                  weight='weight', verbose=verbose,
    #                  super_verbose=False)
    # t02 = time.time()
    # if verbose:
    #     print("Time to run clean_sub_graphs():", t02-t01, "seconds")
    #     print("len G_sub.nodes():", len(G.nodes()))
    #     print("len G_sub.edges():", len(G.edges()))

    # remove self loops
    ebunch = nx.selfloop_edges(G)
    G.remove_edges_from(list(ebunch))

    # save G
    if len(out_gpickle) > 0:
        nx.write_gpickle(G, out_gpickle)

    return G, ske, img_refine

def preprocess(img, thresh, img_mult=255, hole_size=300,
               cv2_kernel_close=7, cv2_kernel_open=7, verbose=False):
    '''
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    hole_size in remove_small_objects is the maximum area, in pixels of the
    hole
    '''
    print(img.size)
    # sometimes get a memory error with this approach
    if img.size < 10000000000:
    # if img.size < 0:
        if verbose:
            print("Run preprocess() with skimage")
        #img = (img > (img_mult * thresh)).astype(np.bool)        
        img = img.astype(np.bool)
        remove_small_objects(img, hole_size, in_place=True)
        remove_small_holes(img, hole_size, in_place=True)
        # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))

    # cv2 is generally far faster and more memory efficient (though less
    #  effective)
    else:
        if verbose:
            print("Run preprocess() with cv2")

        #from road_raster.py, dl_post_process_pred() function
        kernel_close = np.ones((cv2_kernel_close, cv2_kernel_close), np.uint8)
        kernel_open = np.ones((cv2_kernel_open, cv2_kernel_open), np.uint8)
        kernel_blur = cv2_kernel_close
   
        # global thresh
        #mask_thresh = (img > (img_mult * thresh))#.astype(np.bool)
        print("median blur")
        blur = cv2.medianBlur((img * img_mult).astype(np.uint8), kernel_blur)
        print("threshold")
        glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
        glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
        mask_thresh = glob_thresh_arr_smooth      
    
        # opening and closing
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        #gradient = cv2.morphologyEx(mask_thresh, cv2.MORPH_GRADIENT, kernel)
        print("closing")
        closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
        print("opening")
        opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
        img = opening_t.astype(np.bool)
        #img = opening

    return img

def make_skeleton(img_loc, thresh, debug, fix_borders, replicate=5,
                  clip=2, img_shape=(1300, 1300), img_mult=255, hole_size=300,
                  cv2_kernel_close=7, cv2_kernel_open=7,
                  use_medial_axis=False,
                  max_out_size=(200000, 200000),
                  num_classes=1,
                  skeleton_band=7,
                  kernel_blur=27,
                  min_background_frac=0.2,
                  verbose=False):
    '''
    Extract a skeleton from a mask.
    skeleton_band is the index of the band of the mask to use for 
        skeleton extraction, set to string 'all' to use all bands
    '''
    
    if verbose:
        print("Executing make_skeleton...")
    t0 = time.time()
    #replicate = 5
    #clip = 2
    rec = replicate + clip
    weight_arr = None

    
    ds = gdal.Open(img_loc)
    print(ds.RasterYSize)
    print(ds.RasterXSize)
    
    img = ds.ReadAsArray()[skeleton_band,:,:] # only read the first band. second band is predicted orientations
    print(img.shape)
    ds = None
    
    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, 
                                 replicate, cv2.BORDER_REPLICATE)        
#     img_copy = None
#     if debug:
#         if fix_borders:
#             img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
#         else:
#             img_copy = np.copy(img)
    
    t1 = time.time()
    img = preprocess(img, thresh, img_mult=img_mult, hole_size=hole_size,
                     cv2_kernel_close=cv2_kernel_close, 
                     cv2_kernel_open=cv2_kernel_open, verbose=True)
    
    # img, _ = dl_post_process_pred(img)
    
    t2 = time.time()
    if verbose:
        print("Time to run preprocess():", t2-t1, "seconds")
    if not np.any(img): # return None, None
        return None, None
    
    if not use_medial_axis:
        if verbose:
            print("skeletonize...")
        ske = skeletonize(img).astype(np.uint16)
        t3 = time.time()
        if verbose:
            print("Time to run skimage.skeletonize():", t3-t2, "seconds")

    else:
        if verbose:
            print("running updated skimage.medial_axis...")
        ske = skimage.morphology.medial_axis(img).astype(np.uint16)
        t3 = time.time()
        if verbose:
            print("Time to run skimage.medial_axis():", t3-t2, "seconds")

    if fix_borders:
        if verbose:
            print("fix_borders...")
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
        # ske = ske[replicate:-replicate,replicate:-replicate]  
        img = img[replicate:-replicate,replicate:-replicate]
        t4 = time.time()
        if verbose:
            print("Time fix borders:", t4-t3, "seconds")
    
    t1 = time.time()
    if verbose:
        print("ske.shape:", ske.shape)
        print("Time to run make_skeleton:", t1-t0, "seconds")
    
    #print("make_skeletion(), ske.shape:", ske.shape)
    #print("make_skeletion(), ske.size:", ske.size)
    #print("make_skeletion(), ske dtype:", ske.dtype)
    #print("make_skeletion(), ske unique:", np.unique(ske))
    #return
    return img, ske

def write_geotiff(arr, reference_filename, outfilename):
    refds = gdal.Open(reference_filename)
    geotran = refds.GetGeoTransform()
    xres = geotran[1]
    yres = geotran[5]
    xmin = geotran[0]
    ymax = geotran[3]
    ds_projection_ref = refds.GetProjectionRef()
    refds = None
    
    
    driver = gdal.GetDriverByName('GTiff')
    arrshape = arr.shape
    if len(arrshape) > 2:
        nbands = arrshape[0]
        nrows = arrshape[1]
        ncols = arrshape[2]
        #dtype = ds.GetRasterBand(1).DataType
        out_ds = driver.Create(outfilename, ncols, nrows, nbands, gdal.GDT_UInt16)
    else:
        nbands = 1
        nrows = arrshape[0]
        ncols = arrshape[1]
        out_ds = driver.Create(outfilename, ncols, nrows, nbands, gdal.GDT_Byte)
    
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds_projection_ref)
    
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    n = 0
    while n < nbands:
        outband = out_ds.GetRasterBand(n+1)
        if nbands==1:
            outband.WriteArray(arr)
            outband.SetNoDataValue(0)
        else:
            outband.WriteArray(arr[n])
        outband.FlushCache()
        n+=1
    out_ds = None

def make_out_dirs(out_dir, write_shps, write_graphs, write_csvs, write_skeletons):
    bools = [write_shps, write_graphs, write_csvs, write_skeletons]
    dirnames = ["sknw_shps", "sknw_graphs", "sknw_csvs", "skeletons"]
    outdirs = []
    for i in range(len(bools)):
        output_directory = os.path.join(out_dir, dirnames[i])
        if bools[i]:
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            outdirs.append(output_directory)
        else:
            outdirs.append(None)
    return outdirs
    
if __name__ == "__main__":
    args = parse_args()
    img_loc = args.image_filename

    im_dir = args.im_dir
    out_dir = args.out_dir
    write_shps = args.write_shps
    write_graphs = args.write_graphs
    write_csvs = args.write_csvs
    write_skeletons = args.write_skeletons

    if not os.path.exists(im_dir):
        images = [img_loc]
    else:
        #images = [os.path.join(im_dir, n) for n in os.listdir(im_dir) if "roadspeedpred.tif" in n]
        images = glob.glob(im_dir + "/*roadspeedpred.tif")
    
    if out_dir is not None:
        outdirs = make_out_dirs(out_dir, write_shps, write_graphs, write_csvs, write_skeletons)
    else: # just store next to the input images. 
        out_dir = os.path.dirname(images[0])
    outdirs = make_out_dirs(out_dir, write_shps, write_graphs, write_csvs, write_skeletons)
    outshpdir = outdirs[0]
    outgdir = outdirs[1]
    outcsvsdir = outdirs[2]
    outskedir = outdirs[3]
    print("number of prediction images to vectorize: ", len(images))
    print("output shapefile dir: ", outshpdir)
    print("output graph dir: ", outgdir)
    print("output csv dir: ", outcsvsdir)
    print("output skeleton dir: ", outskedir)
    
    final_df = pd.DataFrame(columns=['ImageId', 'WKT_Pix'])
    for img_loc in images:
        # get the relative filepaths of the output
        if outshpdir is not None:
            out_shp = os.path.join(outshpdir, os.path.basename(img_loc)[:-4]+".shp")
        else:
            out_shp = None
        if outgdir is not None:
            out_graph_filename = os.path.join(outgdir, os.path.basename(img_loc)[:-4]+".gpickle")
        else:
            out_graph_filename = ""
        if outcsvsdir is not None:
            out_csv = os.path.join(outcsvsdir, os.path.basename(img_loc)[:-4]+".csv")
        else:
            out_csv = None
        if outskedir is not None:
            out_skeleton_filename = os.path.join(outskedir, os.path.basename(img_loc)[:-4]+"_ske.tif")
        else:
            out_skeleton_filename = None

        img, ske = make_skeleton(img_loc, thresh=0.3, debug=False, 
                                    fix_borders=False, replicate=5,
                                    clip=2, img_shape=(1300, 1300), img_mult=255, hole_size=300,
                                    cv2_kernel_close=7, cv2_kernel_open=7,
                                    use_medial_axis=False,
                                    max_out_size=(200000, 200000),
                                    num_classes=1,
                                    skeleton_band=7,
                                    kernel_blur=27,
                                    min_background_frac=0.2,
                                    verbose=True)

        min_spur_length_m = 10
        ds = gdal.Open(img_loc)
        nrows, ncols = ds.RasterYSize, ds.RasterXSize
        gsd = ds.GetGeoTransform()[1]
        #min_spur_length_pix = int(np.rint(min_spur_length_m / gsd))
        
        min_spur_length_pix = 20

        if ske is None: # ske is None when, preprocess() called within make_skeleton, returns an image without any road detections in the mask
            #return [linestring.format("EMPTY"), [], []]
            ske = np.zeros(shape=(nrows,ncols))
            if out_skeleton_filename is not None:
                write_geotiff(ske, img_loc, out_skeleton_filename)
        else:
            if out_skeleton_filename is not None:
                write_geotiff(ske, img_loc, out_skeleton_filename)

            G, ske, img_refine = img_to_ske_G(img, ske,
                                                min_spur_length_pix=min_spur_length_pix,
                                                out_gpickle=out_graph_filename,
                                                verbose=True)


            df = build_wkt_linestrings(G,
                                        out_csv,
                                        out_shp,
                                        reference_image_filename=img_loc,
                                        add_small=True,
                                        verbose=True,
                                        super_verbose=False)
            final_df = final_df.append(df)
    final_df.to_csv(os.path.join(out_dir, "sknw_wkt.csv"), index=False)