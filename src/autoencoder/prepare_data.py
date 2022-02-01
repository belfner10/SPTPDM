from src.CreateImage.translate_projection import gdal_translate_window
from src.CreateImage.create_image import render_file
import os
import tqdm
import multiprocessing as mp
raster = '../CreateImage/land_cover_data/nlcd_2019_land_cover_l48_20210604.tif'
save_folder = 'data'

xmin = -2493045
xmax = 2342655
ymin = 177285
ymax = 3310005
resolution = 30  # meters per pixel
cell_size = 200  # size of created cells


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



def get_cell_args():
    cells = []
    total_size = cell_size * resolution
    print(f'Number of cells: {int((xmax - xmin) / total_size)*int((ymax - ymin) / total_size)}')
    for x in range(16, int((xmax - xmin) / total_size)):
        for y in range(0, int((ymax - ymin) / total_size)):
            bbox = [str(xmin + total_size * x), str(ymin + total_size * y + total_size), str(xmin + total_size * x + total_size), str(ymin + total_size * y)]
            file_stem = f'{x}_{y}'
            cells.append((file_stem,bbox))
    return cells

if __name__ == '__main__':
    cells = get_cell_args()
    import time
    os.makedirs(save_folder,exist_ok=True)
    s = time.perf_counter()
    with mp.Pool(10) as p:
        results = list(tqdm.tqdm(p.imap_unordered(render_cell, cells, chunksize=100), total=len(cells)))
        x = results

    print(time.perf_counter() - s)



