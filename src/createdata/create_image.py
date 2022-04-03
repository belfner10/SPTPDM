import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from osgeo import gdal

from src.segmenter.funcs import adv_simplify


def get_raster_as_arr(file_path: str):
    raster = gdal.Open(file_path, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr


def get_grey_color_map(file_path: str):
    tree = ET.parse(f'{file_path}.aux.xml')
    root = tree.getroot()
    items = root.findall('PAMRasterBand/GDALRasterAttributeTable/Row')
    hold = []
    for x, item in enumerate(items):
        i = item.findall('F')
        if i[4].text:
            hold.append(x)
            # print(i[4].text)
    h = {t: x for x, t in zip(range(len(hold)), hold)}
    g_lookup = {}
    for x in range(256):
        if x in h:
            g_lookup[x] = h[x]
        else:
            g_lookup[x] = 0
    g_l = lambda x: g_lookup[x]
    return g_l, g_lookup


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


def map_grey_colors(img_arr, color_map):
    img_arr = np.vectorize(color_map)(img_arr).astype(np.uint8)
    return img_arr


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


def render_file(file_path: str, save_path: str, do_simp=False, min_size=10) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} not found')
    if not os.path.exists(file_path + '.aux.xml'):
        raise FileNotFoundError(f'{file_path}.aux.xml not found')

    img_arr = get_raster_as_arr(file_path)
    if do_simp:
        img_arr = adv_simplify(img_arr, min_size)
    if not np.any(img_arr):
        print('Image is all black. Not saving.')
        return
    color_maps = get_color_maps(file_path)
    img_arr2 = map_colors(img_arr, color_maps)

    img = Image.fromarray(img_arr2)
    img.save(save_path)


def render_grey_file(file_path: str, save_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} not found')
    if not os.path.exists(file_path + '.aux.xml'):
        raise FileNotFoundError(f'{file_path}.aux.xml not found')
    img_arr = get_raster_as_arr(file_path)
    if not np.any(img_arr):
        return
    color_map = get_grey_color_map(file_path)
    img_arr2 = map_grey_colors(img_arr, color_map)

    img = Image.fromarray(img_arr2)
    img.save(save_path)


def render_simplified(arr, save_path: str):
    color_map = {0: (0, 0, 0), 1: (84, 117, 168), 2: (181, 0, 0), 3: (210, 205, 192), 4: (56, 129, 78),
                 5: (175, 150, 60), 7: (253, 233, 170), 8: (251, 246, 93), 9: (100, 179, 213)}
    r_lookup = {x: y[0] for x, y in color_map.items()}
    g_lookup = {x: y[1] for x, y in color_map.items()}
    b_lookup = {x: y[2] for x, y in color_map.items()}

    r_l = lambda x: r_lookup[x]
    g_l = lambda x: g_lookup[x]
    b_l = lambda x: b_lookup[x]

    img_arr = arr.astype(np.uint8)
    img_r = np.vectorize(r_l)(img_arr)
    # print('R done')
    img_g = np.vectorize(g_l)(img_arr)
    # print('G done')
    img_b = np.vectorize(b_l)(img_arr)
    img = np.stack((img_r, img_g, img_b), axis=2).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(save_path)


if __name__ == '__main__':
    render_file('temp.tif', 'out.png')
