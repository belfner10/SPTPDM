import sys

sys.path.append("..")

import os
import sys
from os.path import join
from random import choice, randrange, choices, sample

import numpy as np
import scipy.sparse as sp
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from src.clustering.clustering import draw_clusters
from src.createdata.create_image import get_raster_as_arr, save_img
from src.createdata.prepare_data import raster, resolution, format_bbox
from src.createdata.translate_projection import gdal_translate_window, render_file
from src.models.grarep.grarep import get_grarep_comps
from src.models.lae.lae import get_lae_comps
from src.models.line.line import get_line_comps
from src.segmenter.funcs import create_np_adj_from_image, adv_simplify, get_border_counts
from scipy.stats import describe

# there are 21 unique classes in the data
codes = [11, 12, 21, 22, 23, 24, 31, 32, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]

num_classes = 17
output_dir = '.'
file_stem = 'tmp'


# import tensorflow as tf
#
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)


def get_neighbors(adj, node_id):
    return np.where(adj[node_id].toarray() > 0)


def create_seg(adj, comps, tr, avalible):
    seed = choice(avalible)
    print(adj)
    seed, get_neighbors(adj, seed)
    border = set()
    checked = set()
    added = set()


def create_dist_pics(arr):
    labeled, c = label(arr, return_num=True)
    print(c)
    counts, areas = get_border_counts(labeled)

    props = regionprops(labeled)
    area = [prop['area'] for prop in props]
    area.sort()
    print(area[int(len(area) * .8)])
    print(area[-1])
    # area = counts
    print(describe(area))

    fig = plt.gcf()
    dpi = 150
    size = 1250
    fig.set_size_inches(size / dpi, size / dpi)
    plt.hist(area, bins=150)
    plt.ylabel('Counts')
    plt.xlabel('Patch Size')
    plt.yscale('log')
    plt.savefig('size_dist.png', dpi=dpi)
    plt.show()

    plt.clf()
    fig = plt.gcf()
    dpi = 150
    size = 1250
    fig.set_size_inches(size / dpi * 1.05, size / dpi)
    plt.scatter(areas, counts, s=[3] * len(area))
    plt.ylabel('Degree')
    plt.xlabel('Patch Size')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('degree_dist.png', dpi=dpi)
    plt.show()


def get_arr_from_coords(x, y, width, height, do_render=False, file_name='rendered.png'):
    # x and y are in meters while width and height are in pixels
    width, height = width * resolution, height * resolution
    bbox = format_bbox(x, y, width, height)
    if do_render:
        gdal_translate_window(raster, join(output_dir, file_name), bbox)
        arr, palette = get_raster_as_arr(join(output_dir, file_name))
        os.remove(join(output_dir, file_name + '.aux.xml'))
    else:
        gdal_translate_window(raster, join(output_dir, file_stem + '.png'), bbox)
        arr, palette = get_raster_as_arr(join(output_dir, file_stem + '.png'))
        os.remove(join(output_dir, file_stem + '.png'))
        os.remove(join(output_dir, file_stem + '.png.aux.xml'))
    return arr, palette


def main():
    output_dir = '.'
    x, y = 534029, 773017
    x, y = -1937792, 2037729
    x, y = -60845, 2186521
    file_stem = 'tmp'
    width, height = 2500 * resolution, 2500 * resolution

    bbox = format_bbox(x, y, width, height)
    gdal_translate_window(raster, join(output_dir, file_stem + '.tif'), bbox)
    render_file(join(output_dir, file_stem + '.tif'), 'simp_merge_small_sections.png', do_simp=True, min_size=5)
    arr = get_raster_as_arr(join(output_dir, file_stem + '.tif'))
    # render_simplified(arr // 10, 'simp_super_small_sections.png')

    # create_dist_pics(arr)

    # arr = adv_simplify(arr, 3)

    labeled, num_regions = label(arr, return_num=True)
    print(num_regions)
    # props = regionprops(labeled)
    # print(sorted(Counter([prop['area'] for prop in props]).most_common(1000), key=lambda x: (x[1], -x[0])))
    # print(props[0]['bbox'])

    # max_bbox_dim = 400  # median([max(prop['bbox'][3]-prop['bbox'][1],prop['bbox'][2]-prop['bbox'][0]) for prop in props])
    # print(f'mbd: {max_bbox_dim}')
    # eligable = set([prop['label'] for prop in props if prop['bbox'][2] - prop['bbox'][0] > max_bbox_dim * 2 or prop['bbox'][3] - prop['bbox'][1] > max_bbox_dim * 2])
    # print(len(eligable))
    # labeled = labeled.astype('int64')
    # add = len(props) + 20
    # chunk_width = (labeled.shape[0] // max_bbox_dim) + 1
    # for y in range(labeled.shape[0]):
    #     for x in range(labeled.shape[1]):
    #         if labeled[y, x] in eligable:
    #             labeled[y, x] += ((y // max_bbox_dim) * chunk_width + x // max_bbox_dim) * add
    # adv_simp(labeled, 10)
    # labeled2, num_regions2 = label(labeled, return_num=True)

    # gcm, g_lookup = get_grey_color_map(join(output_dir, file_stem + '.tif'))
    # arr = np.vectorize(gcm)(arr).astype(np.uint8)

    os.remove(join(output_dir, file_stem + '.tif'))
    os.remove(join(output_dir, file_stem + '.tif.aux.xml'))

    label_classes, adj, labeled, ignored = create_np_adj_from_image(arr, verbose=True)

    # c = np.array([label_classes[x] % 50 for x in range(1, 1 + len(label_classes))])
    # encoding = np.eye(num_classes)[c]

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
    print(adj.dtype)
    print(type(adj))
    sp.save_npz(f'adj_{len(label_classes)}', adj)
    exit()
    # print('Done')

    lambda_v = 1
    k = 3
    print(type(sp.csr_matrix(adj)))
    # comps = get_grarep_comps(sp.csr_matrix(adj), k, lambda_v, n_components=3)
    # comps = get_lae_comps(adj)
    # comps = np.hstack([comps, encoding * avg])
    # print(comps.shape)
    np.save('comps', comps)
    # comps = np.load('comps.npy')
    avg = np.average(comps)
    print(avg)
    pca = PCA(n_components=3)

    n = pca.fit_transform(comps)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(n.T[0], n.T[1], n.T[2])
    plt.show()

    # create_seg(adj, comps, .1, range(len(comps)))

    # comps = np.load('comps.npy')
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(comps) + 10))

    for x in range(2, 11):
        print(f'x: {x}')
        draw_clusters(comps, labeled, f'{x}clusters.png', n_clusters=x, method='kmeans', ignored=set())
        print()


def gen_random_colors(num):
    all_colors = []
    for x in range(256):
        for y in range(256):
            for z in range(256):
                all_colors.append((x, y, z))
    vals = sample(all_colors, num)
    return np.array(vals, dtype=np.uint8).reshape((-1, 3))


def new_main():
    lambda_v = 1
    k = 3
    x, y = -545943, 1348938
    width, height = 1000, 1000
    land_cover_arr, palette = get_arr_from_coords(x, y, width, height, do_render=True)
    labeled, num_regions = label(land_cover_arr, return_num=True)
    simplified = adv_simplify(labeled, 4)

    # save simplified
    labeled, num_regions = label(simplified, return_num=True)
    print(num_regions)
    pal2 = gen_random_colors(num_regions)
    save_img('simp.png', labeled - 1, do_map_colors=True, color_map=pal2)

    _, adj, labeled, ignored = create_np_adj_from_image(simplified, verbose=True)

    # comps = get_grarep_comps(sp.csr_matrix(adj), k, lambda_v, n_components=64)
    comps = get_lae_comps(adj, n_components=64, num_epochs=5000, learning_rate=.1)
    # comps = get_line_comps(adj)
    pca = PCA(n_components=3)
    n = pca.fit_transform(comps)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(n.T[0], n.T[1], n.T[2])
    plt.show()

    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(comps) + 10))

    for x in range(2, 11):
        print(f'x: {x}')
        draw_clusters(comps, labeled, f'{x}clusters.png', n_clusters=x, method='kmeans', ignored=set())
        print()


if __name__ == '__main__':
    new_main()
