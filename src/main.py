import os
from os.path import join

from PIL import Image
import numpy as np
import scipy.sparse as sp

from src.clustering.clustering import draw_clusters, draw_labels
from src.createdata.create_image import get_raster_as_arr, get_grey_color_map
from src.createdata.prepare_data import raster, resolution, format_bbox
from src.createdata.translate_projection import gdal_translate_window, render_file
from src.models.grarep.grarep import get_grarep_comps
from src.segmenter.funcs import create_np_adj_from_image, create_dict_adj_from_image, simplify_image

import sys

import networkx as nx
import matplotlib.pyplot as plt

# there are 21 unique classes in the data
codes = [11, 12, 21, 22, 23, 24, 31, 32, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]

num_classes = 17


def main():
    output_dir = '.'
    x, y = 534029, 773017
    # x, y = -1937792,2037729
    file_stem = 'tmp'
    width, height = 500 * resolution, 500 * resolution

    bbox = format_bbox(x, y, width, height)
    gdal_translate_window(raster, join(output_dir, file_stem + '.tif'), bbox)
    render_file(join(output_dir, file_stem + '.tif'), 'landcover.png')
    arr = get_raster_as_arr(join(output_dir, file_stem + '.tif'))
    m, g_lookup = get_grey_color_map(join(output_dir, file_stem + '.tif'))
    os.remove(join(output_dir, file_stem + '.tif'))
    os.remove(join(output_dir, file_stem + '.tif.aux.xml'))
    arr = np.vectorize(m)(arr).astype(np.uint8)
    # arr = simplify_image(arr,'second_max')
    label_classes, adj, labeled = create_np_adj_from_image(arr, verbose=True)

    c = np.array([label_classes[x] for x in range(1, 1 + len(label_classes))])
    encoding = np.eye(num_classes)[c]
    # print(encoding.shape)
    # label_classes, weights, labeled = create_dict_adj_from_image(arr, verbose=True)
    # print(labeled - 1)
    # data = 'id_1,id_2,id_3\n'
    # g = nx.Graph()
    # edges = []
    # for key, value in weights.items():
    #     for k2, v2 in value.items():
    #         if key < k2:
    #             edges.append((key - 1, k2 - 1, round(v2,3)))
    #             data += f'{key - 1},{k2 - 1},{v2}\n'
    # g.add_weighted_edges_from(edges)
    # pos = nx.spring_layout(g)
    # nx.draw_networkx(g, pos, with_labels=True)
    # labels = nx.get_edge_attributes(g, 'weight')
    # nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    # # g = nx.from_numpy_matrix(adj)
    # plt.show()
    # # draw_labels(labeled, len(label_classes), 'labeled_labels.png')
    # print(label_classes)
    # exit()
    #
    # with open('out.csv','w') as outfile:
    #     outfile.write(data)

    # np.save('labeled',labeled)
    # print(adj.shape)
    # sp.save_npz(f'adj_{len(label_classes)}', adj)
    # print('Done')

    lambda_v = 1
    k = 8

    # comps = get_grarep_comps(sp.csr_matrix(adj), k, lambda_v, n_components=2)
    # avg = np.average(comps)
    # print(avg)
    # # comps = np.hstack([comps, encoding * avg])
    # # print(comps.shape)
    # np.save('comps',comps)
    comps = np.load('comps.npy')
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(comps) + 10))

    for x in range(2,11):
        print(f'x: {x}')
        draw_clusters(comps, labeled, f'{x}clusters.png', n_clusters=x, method='hierarchical')
        print()


if __name__ == '__main__':
    main()
