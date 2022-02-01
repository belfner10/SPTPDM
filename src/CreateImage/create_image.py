import fiona
import numpy as np
import os
from osgeo import gdal
from PIL import Image
import xml.etree.ElementTree as ET


# with fiona.open("projected/test.shp", "r") as shapefile:
#     state_indicies = [feature['properties']['NAME'] for feature in shapefile]
#     xs = [pt[0] for pt in shapefile[state_indicies.index('Ohio')]['geometry']['coordinates'][0]]
#     ys = [pt[1] for pt in shapefile[state_indicies.index('Ohio')]['geometry']['coordinates'][0]]
#     print(max(xs), min(xs), max(ys), min(ys))
#     ul = (min(xs), max(ys))
#     lr = (max(xs), min(ys))
#     print(f'-projwin {ul[0]} {ul[1]} {lr[0]} {lr[1]}')


# dataset = gdal.Open('nlcd_2019_land_cover_l48_20210604/out3.tif', gdal.GA_ReadOnly)
# # Note GetRasterBand() takes band no. starting from 1 not 0
# band = dataset.GetRasterBand(1)
# arr = band.ReadAsArray()
# print(arr.shape)
# print(arr.dtype)


# tree = ET.parse('nlcd_2019_land_cover_l48_20210604/out3.tif.aux.xml')
# root = tree.getroot()
# items = root.findall('PAMRasterBand/GDALRasterAttributeTable/Row')
#
# r_lookup = {}
# g_lookup = {}
# b_lookup = {}
# a_lookup = {}
#
# for x, item in enumerate(items):
#     i = item.findall('F')
#     r_lookup[x] = int(i[0].text)
#     g_lookup[x] = int(i[1].text)
#     b_lookup[x] = int(i[2].text)
#     a_lookup[x] = int(i[3].text)
#
# r_l = lambda x: r_lookup[x]
# g_l = lambda x: g_lookup[x]
# b_l = lambda x: b_lookup[x]
# a_l = lambda x: a_lookup[x]
#
# img_r = np.vectorize(r_l)(arr)
# print('R done')
# img_g = np.vectorize(g_l)(arr)
# print('G done')
# img_b = np.vectorize(b_l)(arr)
# print('B done')
# img_a = np.vectorize(a_l)(arr)
# print('A done')
#
# img = np.stack((img_r, img_g, img_b, img_a), axis=2).astype(np.uint8)
# print(img.shape, img.dtype)
#
#
# im = Image.fromarray(img)
# print('Saving Image')
# im.save("filename.png")


def get_raster_as_arr(file_path: str):
    raster = gdal.Open(file_path, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr


def get_color_maps(file_path: str):
    tree = ET.parse(f'{file_path}.aux.xml')
    root = tree.getroot()
    items = root.findall('PAMRasterBand/GDALRasterAttributeTable/Row')

    r_lookup = {}
    g_lookup = {}
    b_lookup = {}
    a_lookup = {}

    for x, item in enumerate(items):
        i = item.findall('F')
        r_lookup[x] = int(i[0].text)
        g_lookup[x] = int(i[1].text)
        b_lookup[x] = int(i[2].text)
        a_lookup[x] = int(i[3].text)

    r_l = lambda x: r_lookup[x]
    g_l = lambda x: g_lookup[x]
    b_l = lambda x: b_lookup[x]
    a_l = lambda x: a_lookup[x]

    return r_l, g_l, b_l, a_l


def map_colors(img_arr, color_maps):
    r_l, g_l, b_l, a_l = color_maps
    img_arr = img_arr.astype(np.uint8)
    img_r = np.vectorize(r_l)(img_arr)
    # print('R done')
    img_g = np.vectorize(g_l)(img_arr)
    # print('G done')
    img_b = np.vectorize(b_l)(img_arr)
    # print('B done')
    img_a = np.vectorize(a_l)(img_arr)
    # print('A done')

    img = np.stack((img_r, img_g, img_b), axis=2).astype(np.uint8)
    return img


def render_file(file_path: str, save_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} not found')
    if not os.path.exists(file_path + '.aux.xml'):
        raise FileNotFoundError(f'{file_path}.aux.xml not found')

    img_arr = get_raster_as_arr(file_path)
    if not np.any(img_arr):
        return
    color_maps = get_color_maps(file_path)
    img_arr2 = map_colors(img_arr, color_maps)

    img = Image.fromarray(img_arr2)
    img.save(save_path)




if __name__ == '__main__':
    render_file('temp.tif', 'out.png')
