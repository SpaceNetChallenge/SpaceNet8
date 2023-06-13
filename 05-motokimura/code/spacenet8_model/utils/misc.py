def get_flatten_classes(config):
    groups = config.Class.groups
    classes = config.Class.classes
    assert set(classes.keys()) == set(groups), (classes, groups)
    assert len(groups) == len(set(groups)), groups
    assert len(classes.keys()) == len(set(classes.keys())), classes

    classes_flatten = []
    for g in groups:
        classes_flatten.extend(classes[g])
    return classes_flatten


def save_array_as_geotiff(array, src_geotiff_path, save_path):
    from osgeo import gdal, osr

    ds = gdal.Open(src_geotiff_path)
    xmin, xres, rowrot, ymax, colrot, yres = ds.GetGeoTransform()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds.GetProjectionRef())
    ds = None

    nchannels, nrows, ncols = array.shape
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(save_path, ncols, nrows, nchannels, gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(nchannels):
        outband = out_ds.GetRasterBand(i+1)
        outband.WriteArray(array[i])
        # outband.SetNoDataValue(0)
        outband.FlushCache()
    out_ds = None
