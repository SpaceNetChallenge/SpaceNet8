import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Merge road and building csvs')
    parser.add_argument('--road-csv', required=True)
    parser.add_argument('--bldg-csv', required=True)
    parser.add_argument('--out', default='solution.csv')
    args = parser.parse_args()
    return args


def main(road_csv, bldg_csv, out_file):
    road =pd.read_csv(road_csv)
    bldg = pd.read_csv(bldg_csv)

    # rename column
    bldg.rename(columns={'Wkt_Pix':'WKT_Pix'}, inplace=True)
    
    # add null cloumns
    bldg['length_m']='null'
    bldg['travel_time_s'] = 'null'

    # select meaningful columns
    road = road[['ImageId', 'Object', 'WKT_Pix', 'Flooded', 'length_m', 'travel_time_s']]
    bldg = bldg[['ImageId', 'Object', 'WKT_Pix', 'Flooded', 'length_m', 'travel_time_s']]

    merged = pd.concat([road, bldg], ignore_index=True)
    merged = merged.fillna('null')

    # Convert 'Null' to 'False'
    merged.loc[merged['Flooded']=='Null', 'Flooded'] = 'False'
    # Convert Python bool type to str format
    merged.loc[merged['Flooded']==False, 'Flooded'] = 'False'
    merged.loc[merged['Flooded']==True, 'Flooded'] = 'True'

    # Drop possible duplicates
    merged = merged.drop_duplicates()

    # Save it as csv format
    merged.to_csv(out_file, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args.road_csv, args.bldg_csv, args.out)
