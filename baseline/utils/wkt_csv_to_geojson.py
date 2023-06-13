import os
import glob
import json
from re import L

import geojson
import pandas
import shapely
from shapely import wkt
from shapely import geometry
from osgeo import osr, gdal, ogr

class Image:

    def __init__(self,
                 filepath: str = None,
                 filename: str = None,
                 type: str = None,
                 rows: int = None,
                 cols: int = None,
                 bands: int = None,
                 cell_size: float = None,
                 source_projection: str = None,
                 source_epsg: int = None,
                 geom=None):
        self.filepath = filepath
        self.filename = filename
        self.type = type
        self.rows = rows
        self.cols = cols
        self.bands = bands
        self.cell_size = cell_size
        self.source_projection = source_projection
        self.source_epsg = source_epsg
        self.geom = geom

        self.geotran = None
        self.array = None
        self.srs = None

    def set_from_filepath(self, filepath: str) -> None:
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        ds = gdal.Open(self.filepath)
        self.rows = ds.RasterYSize
        self.cols = ds.RasterXSize
        self.bands = ds.RasterCount
        self.geotran = ds.GetGeoTransform()
        self.cell_size = self.geotran[1]
        im_srs = ds.GetProjection()
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(im_srs)

        self.array = ds.ReadAsArray()
        ds = None
        srs_wkt = osr.SpatialReference(im_srs)
        self.source_projection = srs_wkt.GetAttrValue('projcs')
        self.source_epsg = int(srs_wkt.GetAttrValue('authority', 1))
        ulx = self.geotran[0]
        uly = self.geotran[3]
        lrx = self.geotran[0] + (self.geotran[1] * self.cols)
        lry = self.geotran[3] + (self.geotran[5] * self.rows)
        geom_string = f"POLYGON(({ulx} {uly},{ulx} {lry},{lrx} {lry},{lrx} {uly},{ulx} {uly}))"
        self.geom = wkt.loads(geom_string)

class WKT:

    def __init__(self,
                 wkt):
        self.wkt = wkt

    def as_geo_coords(self,
                      image: Image):
        shapely_geom = wkt.loads(self.wkt)
        if type(shapely_geom) is geometry.Polygon:
            coords = list(shapely_geom.exterior.coords)
        else:
            coords = list(shapely_geom.coords)

        geo_coords = []
        for i in range(len(coords)):
            pix_x = coords[i][0]
            pix_y = coords[i][1]

            geo_x = image.geotran[0] + (image.geotran[1] * pix_x)
            geo_y = image.geotran[3] + (image.geotran[5] * pix_y)
            geo_coords.append([geo_x, geo_y])

        if type(shapely_geom) is geometry.Polygon:
            geo_geom = geometry.Polygon([[p[0], p[1]] for p in geo_coords])
        else:
            geo_geom = geometry.LineString([[p[0], p[1]] for p in geo_coords])
        geo_wkt = geo_geom.wkt
        return geo_wkt

def normalize_true_false(value):
    if isinstance(value, bool):
        pass
    else:
        value = value.lower()
    to_bool = {"true":"True", "false":"False", "null":None, True:"True", False:"False"}
    return to_bool[value]

def image_id_to_wkts(csv_file):
    out_map = {}
    df = pandas.read_csv(csv_file)
    for idx, row in df.iterrows():
        id = row["ImageId"]
        if id not in out_map:
            out_map[id] = []
        flooded = None
        length_m = None
        travel_time_s = None
        
        if 'Flooded' in row:
            flooded = normalize_true_false(row['Flooded'])
        if 'length_m' in row:
            length_m = row['length_m'] 
        if 'travel_time_s' in row:
            travel_time_s = row['travel_time_s']

        if 'Wkt_Pix' in row:
            wkt_pix = row['Wkt_Pix']
        elif 'WKT_Pix' in row:
            wkt_pix = row['WKT_Pix']
        else:
            raise RuntimeError("oops no wkt_pix column")
        out_map[id].append([wkt_pix, {"Flooded":flooded, "length_m":length_m, "travel_time_s":travel_time_s}])
        #if row["Object"] == "Road":
            #print(out_map[id])
    return out_map


if __name__ == "__main__":

    #csv_file = "/nfs/gist/data/RESFLOW/SpaceNet5/SN8/public-test-scores/06-CP0615.csv"
    image_dir = "/nfs/gist/data/RESFLOW/SpaceNet5/SN8/aws_v6/MysteryCity_Test_Private/PRE-event"
    #out_file = "/nfs/gist/data/RESFLOW/SpaceNet5/SN8/public-test-scores/CP0615.geojson"

    csvs = glob.iglob("/nfs/gist/data/RESFLOW/SpaceNet5/SN8/sn8-top5-private-test-scores/*/final.csv", recursive=True)
    
    for csv_file in csvs:
        #out_file = csv_file[:-4] + ".geojson"
        out_filename = csv_file.split("/")[-2] + ".geojson"
        out_file = os.path.join(os.path.dirname(csv_file), out_filename)
        print(csv_file)
        print(out_file)
        print()
        feature_collection = {"type":"FeatureCollection", "features":[]}
        map = image_id_to_wkts(csv_file)

        for id in map:
            image = Image()
            image.set_from_filepath(os.path.join(image_dir, f"{id}.tif"))

            for record in map[id]:
                if record[0] == "POLYGON EMPTY" or record[0] == "LINESTRING EMPTY":
                    pass
                else:
                    geo_wkt = WKT(record[0]).as_geo_coords(image)
                    props = record[1] 

                    g1 = shapely.wkt.loads(geo_wkt)
                    g2 = shapely.geometry.mapping(g1)

                    feature = {"type":"Feature", "geometry": g2, "properties": props}
                    feature_collection['features'].append(feature)

        json_object = json.dumps(feature_collection)
        with open(out_file, "w") as outfile:
            outfile.write(json_object)