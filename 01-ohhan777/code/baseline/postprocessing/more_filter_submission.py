import argparse
import pandas as pd

def more_filtering(in_csv_path, output_csv_path):
    df = pd.read_csv(in_csv_path)
    out_rows = []

    data_in_images = {}
    for index, row in df.iterrows():
        imageid = row["ImageId"]
        if imageid not in data_in_images:
            data_in_images[imageid] = []
        data_in_images[imageid].append(row["Flooded"] in ["True",True])

    for index, row in df.iterrows():
        imageid = row["ImageId"]
        true_percent = sum(data_in_images[imageid]) / len(data_in_images[imageid]) * 100
        if true_percent <= 40:
            row["Flooded"] = False
        elif true_percent >= 70:
            row["Flooded"] = True
        out_rows.append([row[k] for k in list(row.keys())])

    cols = ['ImageId', 'Object', 'WKT_Pix', 'Flooded', 'length_m', 'travel_time_s']
    df = pd.DataFrame(out_rows, columns=cols)
    df_sol = df[['ImageId','Object','WKT_Pix','Flooded','length_m','travel_time_s']]
    df_sol.columns = ['ImageId','Object','Wkt_Pix','Flooded','length_m','travel_time_s']
    df_sol = df_sol.drop_duplicates(keep=False)
    # save
    df_sol.to_csv(output_csv_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv_path",
                        type=str,
                        required=True)
    parser.add_argument("--output_csv_path",
                        type=str,
                        required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    in_csv_path = args.in_csv_path
    output_csv_path = args.output_csv_path

    more_filtering(in_csv_path, output_csv_path)
