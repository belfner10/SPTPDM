import os
from src.createdata.create_image import render_file
from osgeo import ogr, osr
import subprocess
from tqdm.contrib.concurrent import process_map
import tqdm
import multiprocessing as mp

state_fp = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California', 8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
            18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska',
            32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island',
            45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming', 60: 'American Samoa', 66: 'Guam',
            69: 'Northern Mariana Islands', 72: 'Puerto Rico', 78: 'Virgin Islands'}


def check_and_create_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def create_transform(layer):
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.SetFromUserInput("ESRI:102039")

    spatialRef = layer.GetSpatialRef()
    # from Geometry
    # feature = layer.GetNextFeature()
    # geom = feature.GetGeometryRef()
    # inSpatialRef = geom.GetSpatialReference()
    return osr.CoordinateTransformation(spatialRef, outSpatialRef)


def gdal_translate_window(input_image_path, output_image_path, bounding_box):
    print(' '.join(['gdal_translate', input_image_path, output_image_path, '-projwin'] + bounding_box))
    subprocess.check_call(['gdal_translate', input_image_path, output_image_path, '-projwin'] + bounding_box, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


prefix = 'C:/Users/benel/Documents/Test2/'


def process_county(data):
    state, county_name, bbox = data
    if not os.path.exists(f'{prefix}States/{state}/{county_name}'):
        os.makedirs(f'{prefix}States/{state}/{county_name}')
    subprocess.check_call(['gdal_translate', f'{prefix}States/land_cover_data/nlcd_2019_land_cover_l48_20210604.tif', f'{prefix}States/{state}/{county_name}/{county_name}.tif', '-projwin'] + bbox, stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

    render_file(f'{prefix}States/{state}/{county_name}/{county_name}.tif', f'{prefix}States/{state}/{county_name}/{county_name}.png')


def process_county2(data):
    state, county_name, bbox = data
    check_and_create_path(f'{prefix}States/{state}/{county_name}')
    gdal_translate_window(f'C:/Users/benel/Documents/States/land_cover_data/nlcd_2019_land_cover_l48_20210604.tif', f'{prefix}States/{state}/{county_name}/{county_name}.tif', bbox)
    render_file(f'{prefix}States/{state}/{county_name}/{county_name}.tif', f'{prefix}States/{state}/{county_name}/{county_name}.png')


if __name__ == '__main__':

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(r'../projected/test.shp')

    layer = dataset.GetLayer()

    coordTrans = create_transform(layer)

    counties = []

    for feature in layer:
        m = [f for f in dir(feature) if f[0] != '_']
        county_name = feature.GetField('NAME')
        state = state_fp[int(feature.GetField('STATEFP'))]
        if county_name != 'Black Hawk':
            continue
        # if state != 'Kansas' and state != 'California':
        #     continue
        geom = feature.GetGeometryRef()

        geom.Transform(coordTrans)
        bbox = geom.GetEnvelope()
        bbox = [str(bbox[0]), str(bbox[3]), str(bbox[1]), str(bbox[2])]
        print(' '.join(bbox))
        counties.append((state, county_name, bbox))

    import time

    s = time.perf_counter()
    with mp.Pool(16) as p:
        results = list(tqdm.tqdm(p.imap_unordered(process_county2, counties, chunksize=10), total=len(counties)))
        x = results
    # for data in counties[20:]:
    #     print(data)
    #     process_county(data)

    print(time.perf_counter() - s)
