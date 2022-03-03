import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from skimage.measure import label

from src.clustering.funcs import create_n_clusters
from src.segmenter.funcs import simplify_image

from random import sample
from collections import defaultdict
import matplotlib.pyplot as plt

# colors are from the muted color set at https://personal.sron.nl/~pault/
# colors are in RGB format
colors = [[204, 102, 119], [51, 34, 136], [221, 204, 119], [17, 119, 51], [136, 204, 238], [136, 34, 85], [68, 170, 153], [153, 153, 51], [170, 68, 153], [221, 221, 221], [0, 0, 0]]


def create_color_maps(n_clusters):
    # TODO add check to make sure n_clusters is less than or equal to colors somewhere upstream
    selected_colors = sample(colors, n_clusters)
    rmap = {}
    gmap = {}
    bmap = {}

    for x in range(n_clusters):
        rmap[x] = selected_colors[x][0]
        gmap[x] = selected_colors[x][1]
        bmap[x] = selected_colors[x][2]

    r_l = lambda x: rmap[x]
    g_l = lambda x: gmap[x]
    b_l = lambda x: bmap[x]

    return r_l, g_l, b_l


def invert_dict(d):
    ret = defaultdict(list)
    for key, value in d.items():
        ret[value].append(key)
    return ret


def hierarchical_clustering(comps, labeled, n_clusters, method='ward'):
    Z = linkage(comps, method)
    clusters = create_n_clusters(Z, len(comps), num_clusters=n_clusters)
    labeled_clusters = np.vectorize(lambda x: clusters[x])(labeled)

    # inv = invert_dict(clusters)
    #
    # for key, value in inv.items():
    #     print(len(value))

    return labeled_clusters


def k_means_clustering(comps, labeled, n_clusters):
    kmeans = KMeans(n_clusters).fit(comps)
    clusters = {}
    for x, label in enumerate(kmeans.labels_):
        clusters[x + 1] = label
    labeled_clusters = np.vectorize(lambda c: clusters[c])(labeled)
    return labeled_clusters


def draw_clusters(comps, labeled, filename, n_clusters=5, method='kmeans'):
    if method == 'kmeans':
        labeled_clusters = k_means_clustering(comps, labeled, n_clusters)
    elif method == 'hierarchical':
        labeled_clusters = hierarchical_clustering(comps, labeled, n_clusters)
    else:
        raise ValueError(f'Unknown method name: {method}. Possible values are [\'kmeans\', \'hierarchical\']')

    r_l, g_l, b_l = create_color_maps(n_clusters)
    img_r = np.vectorize(r_l)(labeled_clusters)
    img_g = np.vectorize(g_l)(labeled_clusters)
    img_b = np.vectorize(b_l)(labeled_clusters)
    img = np.stack((img_r, img_g, img_b), axis=2).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def draw_labels(labeled, n_labels, filename):
    r_l, g_l, b_l = create_color_maps(n_labels)
    img_r = np.vectorize(r_l)(labeled-1)
    img_g = np.vectorize(g_l)(labeled-1)
    img_b = np.vectorize(b_l)(labeled-1)
    img = np.stack((img_r, img_g, img_b), axis=2).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    vecs = np.load('out_grarep.np.npy')[:, 1:]

    Z = linkage(vecs, 'complete')

    image = cv2.imread('1000.png', cv2.IMREAD_GRAYSCALE)
    image = simplify_image(image, 'second_max')
    labeled, num_regions = label(image, return_num=True)
    print('Creating Images')
    for r in range(2, 10):
        print(f'Creating Image {r}')
        m = create_n_clusters(Z, len(vecs), r)
        mult = int(250 / r)
        out = np.zeros(labeled.shape, dtype='uint8')
        for y in range(labeled.shape[0]):
            for x in range(labeled.shape[1]):
                out[y, x] = m[labeled[y, x]] * mult
        cv2.imwrite(f'out{r}.png', out)


def main2():
    comps = np.load('comps.npy')
    labeled = np.load('labeled.npy')
    print(comps.shape)
    draw_clusters(comps, labeled, n_clusters=10)


if __name__ == '__main__':
    main2()
