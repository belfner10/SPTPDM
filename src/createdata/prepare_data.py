from src.CreateImage.translate_projection import gdal_translate_window
from src.CreateImage.create_image import render_file, render_grey_file, get_raster_as_arr, get_grey_color_map
import os
import numpy as np
import tqdm
import multiprocessing as mp
import gzip
from PIL import Image

raster = 'land_cover_data/nlcd_2019_land_cover_l48_20210604.tif'
save_folder = 'data'

xmin = -2493045
xmax = 2342655
ymin = 177285
ymax = 3310005
width = xmax - xmin
height = ymax - ymin
# print(f'Height: {height}\nWidth: {width}')
resolution = 30  # meters per pixel
cell_size = 100  # size of created cells
cells_height = int(height / resolution / cell_size + 1)
cells_width = int(width / resolution / cell_size + 1)


def get_cell_at_point(x, y):
    x_offset = x - xmin
    cell_x = int(x_offset / resolution / cell_size)
    y_offset = y - ymin
    cell_y = int(y_offset / resolution / cell_size)
    return cell_x, cell_y


# not complete
# def get_img_around_point(x,y,size):

#     if size % 2 == 0:
#         size *=2
#         bbox = [str(x-size/2),str(y+size/2),str(x+size/2),str(y-size/2)]
#     else:
#         size*=2
#         bbox = [str(x-size/2+15),str(y+size/2),str(x+size/2),str(y-size/2)]


def render_cell(args):
    file_stem, bbox = args
    gdal_translate_window(raster, os.path.join(save_folder, file_stem + '.tif'), bbox)
    render_file(os.path.join(save_folder, file_stem + '.tif'), os.path.join(save_folder, file_stem + '.png'))
    while os.path.exists(os.path.join(save_folder, file_stem + '.tif')):
        try:
            os.remove(os.path.join(save_folder, file_stem + '.tif'))
        except PermissionError:
            pass
    while os.remove(os.path.join(save_folder, file_stem + '.tif.aux.xml')):
        try:
            os.remove(os.path.join(save_folder, file_stem + '.tif.aux.xml'))
        except PermissionError:
            pass


def render_grey_cell(args):
    file_stem, bbox = args
    gdal_translate_window(raster, os.path.join(save_folder, file_stem + '.tif'), bbox)
    render_grey_file(os.path.join(save_folder, file_stem + '.tif'), os.path.join(save_folder, file_stem + '.png'))
    while os.path.exists(os.path.join(save_folder, file_stem + '.tif')):
        try:
            os.remove(os.path.join(save_folder, file_stem + '.tif'))
        except PermissionError:
            pass
    while os.remove(os.path.join(save_folder, file_stem + '.tif.aux.xml')):
        try:
            os.remove(os.path.join(save_folder, file_stem + '.tif.aux.xml'))
        except PermissionError:
            pass


def get_single_cell_args(x, y):
    total_size = cell_size * resolution
    bbox = [str(xmin + total_size * x), str(ymin + total_size * y + total_size), str(xmin + total_size * x + total_size), str(ymin + total_size * y)]
    file_stem = f'{x}_{y}'
    return file_stem, bbox


def get_cell_args():
    cells = []
    total_size = cell_size * resolution
    print(f'Number of cells: {int((xmax - xmin) / total_size) * int((ymax - ymin) / total_size)}')
    for x in range(0, int((xmax - xmin) / total_size)):
        for y in range(0, int((ymax - ymin) / total_size)):
            bbox = [str(xmin + total_size * x), str(ymin + total_size * y + total_size), str(xmin + total_size * x + total_size), str(ymin + total_size * y)]
            file_stem = f'{x}_{y}'
            cells.append((file_stem, bbox))
    return cells


def format_bbox(x: int, y: int, width: int, height: int) -> [str]:
    """
    Formats bounding box term for raster calculations (x and y are the lower right corner of the image)
    Parameters
    ----------
    x : int
    y : int
    width : int
    height : int

    Returns
    -------

    """
    # align bbox to cell boarders
    x -= (x-xmin) % 30
    y -= (y-ymin) % 30
    width -= width % 30
    height -= height % 30

    if width == 0 or height == 0:
        raise ValueError(f'Specified area too small')

    bbox = [str(int(x)), str(int(y + height)), str(int(x + width)), str(int(y))]
    return bbox





if __name__ == '__main__':
    save_folder = 'd2'
    os.makedirs(save_folder, exist_ok=True)
    x, y = -1082535, 1566255
    file_stem = 's'
    width, height = 100 * resolution, 50 * resolution
    bbox = format_bbox(x, y, width, height)
    print(bbox)

    # gdal_translate_window(raster, d, bbox)
    # x, y = 722262.4, 1748953.9
    # cell_x, cell_y = get_cell_at_point(x, y)
    # print(cell_x, cell_y)
    # cell = get_single_cell_args(cell_x, cell_y)
    # file_stem, bbox = cell
    # print(bbox,file_stem)
    # render_cell(cell)
    gdal_translate_window(raster, os.path.join(save_folder, file_stem + '.tif'), bbox)
    arr = get_raster_as_arr(os.path.join(save_folder, file_stem + '.tif'))
    m, g_lookup = get_grey_color_map(os.path.join(save_folder, file_stem + '.tif'))
    arr = np.vectorize(m)(arr).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(os.path.join(save_folder, file_stem + 'g.png'))
