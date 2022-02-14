from src.CreateImage.translate_projection import gdal_translate_window
from src.CreateImage.create_image import render_file, render_grey_file, get_raster_as_arr, get_grey_color_map
import os
import numpy as np
import tqdm
import multiprocessing as mp
import gzip
from PIL import Image

raster = '../CreateImage/land_cover_data/nlcd_2019_land_cover_l48_20210604.tif'
save_folder = 'data'

xmin = -2493045
xmax = 2342655
ymin = 177285
ymax = 3310005
width = xmax - xmin
height = ymax - ymin
# print(f'Height: {height}\nWidth: {width}')
resolution = 30  # meters per pixel
cell_size = 500  # size of created cells
cells_height = int(height / resolution / cell_size + 1)
cells_width = int(width / resolution / cell_size + 1)


# print(f'Cells hight: {cells_height}\nCells width: {cells_width}')


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


if __name__ == '__main__':

    # cells = get_cell_args()
    # import time
    #
    # print(len(cells))
    #
    # s = time.perf_counter()
    # with mp.Pool(10) as p:
    #     results = list(tqdm.tqdm(p.imap_unordered(render_cell, cells, chunksize=10), total=len(cells)))
    #     _ = results
    #
    # print(time.perf_counter() - s)
    save_folder = 'd2'
    os.makedirs(save_folder, exist_ok=True)
    x, y = 722262.4,1748953.9
    cell_x, cell_y = get_cell_at_point(x, y)
    print(cell_x, cell_y)
    cell = get_single_cell_args(cell_x, cell_y)
    file_stem, bbox = cell
    render_cell(cell)
    gdal_translate_window(raster, os.path.join(save_folder, file_stem + '.tif'), bbox)
    arr = get_raster_as_arr(os.path.join(save_folder, file_stem + '.tif'))
    m, g_lookup = get_grey_color_map(os.path.join(save_folder, file_stem + '.tif'))
    arr = np.vectorize(m)(arr).astype(np.uint8)
    l = max(g_lookup.values())
    arr2 = np.eye(l + 1)[arr].astype('b') # 1 hot encoding of classes
    # print(arr2)
    print(arr2.nbytes)
    red = np.packbits(arr2, axis=None)
    # print(red)
    print(red.nbytes)
    unpacked = np.unpackbits(red, count=arr2.size).reshape(arr2.shape).view('b')
    print(np.prod(arr2.shape), arr2.size)
    # print(unpacked)
    print(np.array_equal(arr2, unpacked))
    # with open(,'wb') as outfile:
    np.save(os.path.join(save_folder, file_stem + '+' + str(arr2.shape) + 'l.npy'), arr2)
    np.save(os.path.join(save_folder, file_stem + '+' + str(arr2.shape) + '.npy'), red)
    f = gzip.GzipFile(os.path.join(save_folder, file_stem + '+' + str(arr2.shape) + '.npy.gz'), 'wb')
    np.save(f, red)
    f.close()
    img = Image.fromarray(arr)
    img.save(os.path.join(save_folder, file_stem + 'g.png'))

    f = gzip.GzipFile(os.path.join(save_folder, file_stem + '+' + str(arr2.shape) + '.npy.gz'), 'rb')
    arr3 = np.unpackbits(np.load(f), count=arr2.size).reshape(arr2.shape).view('b')
    f.close()
    print(np.array_equal(arr2, arr3))
    # render_grey_cell(cell)
    # render_cell(cell)
