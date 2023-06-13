import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# isort: off
from spacenet8_model.utils.postproc_road_to_graph import wkt_to_G
# isort: on


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector', required=True)
    parser.add_argument('--min_subgraph_length_pix',
                        type=int,
                        default=20)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    vector_csv_path = os.path.join(args.vector, 'road_vectors.csv')
    df = pd.read_csv(vector_csv_path)
    image_ids = np.sort(np.unique(df['ImageId']))[:]  # XXX: actually image paths

    # TODO
    min_subgraph_length_pix = args.min_subgraph_length_pix
    simplify_graph = True
    verbose = False
    pickle_protocol = 4     # 4 is most recent, python 2.7 can't read 4
    node_iter = 10000  # start int for node naming
    edge_iter = 10000  # start int for edge naming
    manually_reproject_nodes = False
    rdp_epsilon = 1

    # XXX: variables below are not used
    min_spur_length_m = 10
    n_threads = 1

    out_root = os.path.basename(os.path.normpath(args.vector))
    if args.val:
        out_root = os.path.join(args.artifact_dir, '_val/road_graphs', out_root)
    else:
        out_root = os.path.join(args.artifact_dir, 'road_graphs', out_root)
    print(f'going to save road graphs under {out_root}')

    params = []
    for i, image_id in enumerate(image_ids):
        aoi = os.path.basename(os.path.dirname(image_id))
        out_dir = os.path.join(out_root, aoi)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{os.path.basename(image_id).split('.')[0]}.gpickle")
 
        wkt_list = df['WKT_Pix'][df['ImageId'] == image_id].values
        
        if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
            G = nx.MultiDiGraph()
            nx.write_gpickle(G, out_file, protocol=pickle_protocol)
            continue
        else:
            params.append((
                wkt_list, image_id, min_subgraph_length_pix,
                node_iter, edge_iter,
                min_spur_length_m, simplify_graph,
                rdp_epsilon,
                manually_reproject_nodes, 
                out_file, out_dir, n_threads, verbose
            ))

    # execute
    for i in tqdm(range(len(params))):
        wkt_to_G(params[i])

    print(f'saved under {out_root}')


if __name__ == '__main__':
    main()
