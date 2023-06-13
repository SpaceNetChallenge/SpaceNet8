import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import fire
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.wkt
import skimage.io
from osgeo import ogr, osr
from shapely.geometry import LineString, MultiPoint, Point

from sn8.evaluation.length_apls import cut_linestring_by_distance
from sn8.post_processing.roads import infer_speed, vectorize_roads, wkt_to_G


def run_vectorize_roads(
    mask_path: str, out_prefix: str, min_spur_length_pix: int = 20, thresh: float = 0.3
) -> Sequence[str]:
    img, ske = vectorize_roads.make_skeleton(
        mask_path,
        thresh=thresh,
        debug=False,
        fix_borders=False,
        replicate=5,
        clip=2,
        img_shape=(1300, 1300),
        img_mult=255,
        hole_size=300,
        cv2_kernel_close=7,
        cv2_kernel_open=7,
        use_medial_axis=False,
        max_out_size=(200000, 200000),
        num_classes=1,
        skeleton_band=7,
        kernel_blur=27,
        min_background_frac=0.2,
        verbose=True,
    )

    # TODO: ske の保存

    if ske is None:
        return []
    else:
        G, ske, _ = vectorize_roads.img_to_ske_G(
            img,
            ske,
            min_spur_length_pix=min_spur_length_pix,
            out_gpickle=f"{out_prefix}_G_pix.gpickle",
            verbose=True,
        )

        wkt_list = vectorize_roads.G_to_wkt(G, add_small=True, verbose=True, super_verbose=False)
        return wkt_list


def run_wkt_to_G(
    mask_path: str,
    out_prefix: str,
    wkt_list: Sequence[str],
    min_subgraph_length_pix: int,
    min_spur_length_m: int,
) -> nx.MultiDiGraph:
    simplify_graph = True  # True # False
    verbose = False  # sFalse
    node_iter = 10000  # start int for node naming
    edge_iter = 10000  # start int for edge naming
    manually_reproject_nodes = False  # True
    n_threads = 1
    rdp_epsilon = 0  # 1 だと geometry_latlon_wkt への RDP が変な挙動をする && 前段階で approxPolyDP である程度近似している ので 0 にする

    # global な logger を `if __name__ == "__main__"` の中で setup しているので、外から作ってあげる必要がある
    logger = logging.getLogger(wkt_to_G.__name__)
    wkt_to_G.logger1 = logger

    G = wkt_to_G.wkt_to_G(
        (
            wkt_list,
            mask_path,
            min_subgraph_length_pix,
            node_iter,
            edge_iter,
            min_spur_length_m,
            simplify_graph,
            rdp_epsilon,
            manually_reproject_nodes,
            f"{out_prefix}_G_utm.gpickle",
            None,
            n_threads,
            verbose,
        )
    )
    return G


def run_infer_speed(
    mask_path: str, out_prefix: str, G_raw: nx.MultiDiGraph, speed_conversion_file: str
) -> nx.MultiDiGraph:
    pickle_protocol = 4  # 4 is most recent, python 2.7 can't read 4

    mask = skimage.io.imread(mask_path)
    mask = np.moveaxis(mask, -1, 0)

    # see if it's empty
    if len(G_raw.nodes()) == 0:
        return G_raw

    skeleton_band = 7
    max_speed_band = skeleton_band - 1
    percentile = 85  # percentil filter (default = 85)
    dx, dy = 6, 6  # nearest neighbors patch size  (default = (4, 4))
    min_z = 128  # min z value to consider a hit (default = 128)
    mph_to_mps = 0.44704  # miles per hour to meters per second
    use_weighted_mean = True
    variable_edge_speed = False
    verbose = False

    # global な logger を `if __name__ == "__main__"` の中で setup しているので、外から作ってあげる必要がある
    logger = logging.getLogger(infer_speed.__name__)
    infer_speed.logger1 = logger

    _, conv_dict = infer_speed.load_speed_conversion_dict_binned(speed_conversion_file)

    for _, (_, _, edge_data) in enumerate(G_raw.edges(data=True)):
        tot_hours, mean_speed_mph, length_miles = infer_speed.get_edge_time_properties(
            mask,
            edge_data,
            conv_dict,
            min_z=min_z,
            dx=dx,
            dy=dy,
            percentile=percentile,
            max_speed_band=max_speed_band,
            use_weighted_mean=use_weighted_mean,
            variable_edge_speed=variable_edge_speed,
            verbose=verbose,
        )
        # update edges
        edge_data["Travel Time (h)"] = tot_hours
        edge_data["inferred_speed_mph"] = np.round(mean_speed_mph, 2)
        edge_data["length_miles"] = length_miles
        edge_data["inferred_speed_mps"] = np.round(mean_speed_mph * mph_to_mps, 2)
        edge_data["travel_time_s"] = np.round(3600.0 * tot_hours, 3)
        # edge_data['travel_time'] = np.round(3600. * tot_hours, 3)

    G = G_raw.to_undirected()
    nx.write_gpickle(G, f"{out_prefix}_G_speed.gpickle", protocol=pickle_protocol)
    return G


def make_wkt_df(G: nx.MultiDiGraph) -> pd.DataFrame:
    columns = ["Object", "WKT_Pix", "WKT_Geo", "length_m", "travel_time_s"]

    if len(G.nodes()) == 0:
        return pd.DataFrame([["Road", "LINESTRING EMPTY", "LINESTRING EMPTY", 0, 0]], columns=columns)

    rows = []
    seen_edges = set([])
    for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):
        # make sure we haven't already seen this edge
        if (u, v) in seen_edges or (v, u) in seen_edges:
            continue
        else:
            seen_edges.add((u, v))
            seen_edges.add((v, u))

        geom_pix = attr_dict["geometry_pix"]
        if type(geom_pix) != str:
            geom_pix_wkt = attr_dict["geometry_pix"].wkt
        else:
            geom_pix_wkt = geom_pix

        geom_geo = attr_dict["geometry_utm_wkt"].wkt
        geom = ogr.CreateGeometryFromWkt(geom_geo)

        targetsrs = osr.SpatialReference()
        targetsrs.ImportFromEPSG(4326)

        utm_zone = attr_dict["utm_zone"]
        source = osr.SpatialReference()  # the input dataset is in wgs84
        source.ImportFromProj4(f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs")
        transform_to_utm = osr.CoordinateTransformation(source, targetsrs)
        geom.Transform(transform_to_utm)
        geom_geo_wkt = geom.ExportToWkt()

        rows.append(["Road", geom_pix_wkt, geom_geo_wkt, attr_dict["length"], attr_dict["travel_time_s"]])

    return pd.DataFrame(rows, columns=columns)


def insert_flood_pred(df: pd.DataFrame, flood_mask: str, flood_th: float) -> None:
    dy = 2
    dx = 2

    flood = None

    is_flooded = []
    for wkt_pix in df["WKT_Pix"]:
        if wkt_pix == "LINESTRING EMPTY":
            is_flooded.append(False)
        else:
            if flood is None:
                flood = skimage.io.imread(flood_mask)  # {0, ..., 255}^n
                flood = ((flood / 255) >= flood_th).astype(np.uint8)  # {0, 1}^n
                assert flood.ndim == 2
                nrows, ncols = flood.shape

            geom = ogr.CreateGeometryFromWkt(wkt_pix)

            nums = []  # flood prediction vals
            for i in range(0, geom.GetPointCount() - 1):
                pt1 = geom.GetPoint(i)
                pt2 = geom.GetPoint(i + 1)
                dist = math.ceil(math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2))
                x0, y0 = pt1[0], pt1[1]
                x1, y1 = pt2[0], pt2[1]
                x, y = np.linspace(x0, x1, dist).astype(int), np.linspace(y0, y1, dist).astype(int)

                for i in range(len(x)):
                    top = max(0, y[i] - dy)
                    bot = min(nrows - 1, y[i] + dy)
                    left = max(0, x[i] - dx)
                    right = min(ncols - 1, x[i] + dx)
                    nums.extend(flood[top:bot, left:right].flatten())

            is_flooded.append(np.argmax(np.bincount(nums)) == 1)

    df["Flooded"] = is_flooded


def extend_near_junction_road(df: pd.DataFrame, extension_length_pix: float) -> gpd.GeoSeries:
    roads = gpd.GeoSeries(df["WKT_Pix"].map(shapely.wkt.loads)).copy()

    for side in ["start", "end"]:
        for i, road in enumerate(roads):
            if len(road.coords) < 2:
                print("Unexpected road:", road)
                continue
            coords = np.asarray(road.coords)

            if side == "start":
                p = coords[0]
                vec = p - coords[1]
            else:
                p = coords[-1]
                vec = p - coords[-2]

            distances = roads.distance(Point(p))
            for j in np.argsort(distances):
                if i != j:
                    break
            if i == j:
                continue
            if distances[j] == 0.0:
                continue

            vec = vec / (np.sqrt(vec[0] ** 2 + vec[1] ** 2) + 1e-9) * extension_length_pix
            new_p = p + vec

            intersect_geom = roads.iloc[j].intersection(LineString([new_p, p]))
            if isinstance(intersect_geom, MultiPoint):
                intersection = np.concatenate([np.asarray(p.coords) for p in intersect_geom.geoms], axis=0)
            else:
                try:
                    intersection = np.asarray(intersect_geom.coords)
                except NotImplementedError:
                    print("Unexpected intersection:", intersect_geom)
                    continue

            if len(intersection) > 0:
                new_p = intersection[np.argsort(((new_p - intersection) ** 2).sum(axis=1))[0]]

                road_j = roads.iloc[j]
                coords_j = np.asarray(road_j.coords)
                distances = np.sqrt(((coords_j - new_p) ** 2).sum(axis=1))
                k = np.argmin(distances)
                if distances[k] <= 1:
                    new_p = coords_j[k]
                else:
                    road_j_a, road_j_b = cut_linestring_by_distance(road_j, road_j.project(Point(new_p)))
                    assert road_j_a.coords[-1] == road_j_b.coords[0]
                    new_p = road_j_a.coords[-1]
                    roads.iloc[j] = LineString(list(road_j_a.coords) + list(road_j_b.coords[1:]))

                if side == "start":
                    roads.iloc[i] = LineString([new_p] + list(coords))
                else:
                    roads.iloc[i] = LineString(list(coords) + [new_p])

    df["WKT_Pix"] = [road.wkt for road in roads]


def remove_isolate_small_path(
    df: pd.DataFrame,
    small_isolate_path_length_pix: float = 200,
) -> pd.DataFrame:
    roads = gpd.GeoSeries(df["WKT_Pix"].map(shapely.wkt.loads))

    image_border_pix = 50
    not_removed = []
    for i, line in enumerate(roads):
        ok = not (roads.intersects(line).sum() == 1 and line.length < small_isolate_path_length_pix)

        if not ok:
            for (x, y) in line.coords:
                # FIXME: Avoid hardcoding 1300
                if min(x, y) < image_border_pix or 1300 - image_border_pix <= max(x, y):
                    ok = True
                    break
        if ok:
            not_removed.append(i)
        else:
            print(f"Removed: {line.length}")

    return df.iloc[not_removed]


def main(
    mask_path: str,
    out_dir: str,
    speed_conversion_file: str,
    flood_mask: Optional[str] = None,
    road_skeleton_th: float = 0.3,
    min_spur_length_pix: int = 20,
    min_subgraph_length_pix: int = 20,
    min_spur_length_m: int = 10,
    flood_th: float = 0.5,
    extension_length_pix: Optional[float] = None,
    small_isolate_path_length_pix: Optional[float] = None,
) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    out_prefix = str(Path(out_dir) / (Path(mask_path).stem))

    wkt_list = run_vectorize_roads(mask_path, out_prefix, min_spur_length_pix, road_skeleton_th)
    G_utm = run_wkt_to_G(mask_path, out_prefix, wkt_list, min_subgraph_length_pix, min_spur_length_m)
    G_speed = run_infer_speed(mask_path, out_prefix, G_utm, speed_conversion_file)

    df = make_wkt_df(G_speed)
    if flood_mask is not None:
        insert_flood_pred(df, flood_mask, flood_th)

    if extension_length_pix is not None:
        extend_near_junction_road(df, extension_length_pix)

    if small_isolate_path_length_pix is not None:
        df = remove_isolate_small_path(df, small_isolate_path_length_pix)

    df.to_csv(f"{out_prefix}_wkt_df.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
