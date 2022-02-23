from os.path import join

from PIL import Image
import numpy as np
import scipy.sparse as sp


from src.clustering.clustering import create_several_clustes
from src.createdata.create_image import get_raster_as_arr, get_grey_color_map
from src.createdata.prepare_data import raster, resolution, format_bbox
from src.createdata.translate_projection import gdal_translate_window
from src.models.grarep.grarep import get_k_components
from src.segmenter.funcs import create_np_adj_from_image

output_dir = '.'
x, y = -1082535, 1566255
file_stem = 'tmp'
width, height = 1000 * resolution, 1000 * resolution
bbox = format_bbox(x, y, width, height)
gdal_translate_window(raster, join(output_dir, file_stem + '.tif'), bbox)
arr = get_raster_as_arr(join(output_dir, file_stem + '.tif'))
m, g_lookup = get_grey_color_map(join(output_dir, file_stem + '.tif'))
arr = np.vectorize(m)(arr).astype(np.uint8)
img = Image.fromarray(arr)
img.save(join(output_dir, file_stem + 'g.png'))
label_classes, adj = create_np_adj_from_image(arr, verbose=True)
print('Done')
size = 20
lambda_v = 3
comps = get_k_components(sp.csr_matrix(adj), 3, lambda_v, n_components=4)
# image = create_several_clustes(comps, join(output_dir, file_stem + 'g.png'))
# print(np.array_equal(arr,image))
