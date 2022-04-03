import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans, SpectralClustering
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


def create_color_maps2(seeds):
    # TODO add check to make sure n_clusters is less than or equal to colors somewhere upstream
    selected_colors = [colors[x] for x, _ in seeds]
    rmap = {}
    gmap = {}
    bmap = {}

    for x in range(len(seeds)):
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


def hierarchical_clustering(comps, labeled, n_clusters, method='ward', ignored=set()):
    Z = linkage(comps, method)
    clusters, seeds = create_n_clusters(Z, len(comps), num_clusters=n_clusters)
    labeled_clusters = np.vectorize(lambda x: clusters[x])(labeled)

    # for label in ignored:

    inv = invert_dict(clusters)

    for key, value in inv.items():
        print(len(value))

    return labeled_clusters, seeds


def k_means_clustering(comps, labeled, n_clusters):
    kmeans = KMeans(n_clusters).fit(comps)
    clusters = {}
    for x, label in enumerate(kmeans.labels_):
        clusters[x + 1] = label

    inv = invert_dict(clusters)

    for key, value in inv.items():
        print(len(value))

    labeled_clusters = np.vectorize(lambda c: clusters[c])(labeled)
    return labeled_clusters


def bi_k_means_clustering(comps, labeled, n_clusters):
    bi_comps = np.copy(comps)
    current_clusters = 1
    clusters = {}
    xs = np.array(range(len(comps)))
    temp_clusters = []
    while current_clusters != n_clusters:
        kmeans = KMeans(n_clusters=2).fit(bi_comps)

        cluster_centers = kmeans.cluster_centers_
        sse = [0] * 2
        for point, label in zip(bi_comps, kmeans.labels_):
            sse[label] += np.square(point - cluster_centers[label]).sum()
        chosen_cluster = np.argmax(sse, axis=0)
        if chosen_cluster == 0:
            temp_clusters.append((sse[0], np.array([pt for pt, label in zip(xs, kmeans.labels_) if label == 0])))
            temp_clusters.append((sse[1], np.array([pt for pt, label in zip(xs, kmeans.labels_) if label != 0])))
        else:
            temp_clusters.append((sse[1], np.array([pt for pt, label in zip(xs, kmeans.labels_) if label == 1])))
            temp_clusters.append((sse[0], np.array([pt for pt, label in zip(xs, kmeans.labels_) if label != 1])))

        if current_clusters == n_clusters - 1:
            break

        temp_clusters.sort(key=lambda x: x[0])

        bi_comps = comps[temp_clusters[-1][1]]
        xs = temp_clusters[-1][1]
        temp_clusters.pop(-1)
        current_clusters += 1
    x = 0
    for _, pts in temp_clusters:
        for pt in pts:
            clusters[pt + 1] = x
        x += 1

    inv = invert_dict(clusters)

    for key, value in inv.items():
        print(len(value))

    labeled_clusters = np.vectorize(lambda c: clusters[c])(labeled)
    return labeled_clusters


def spectral_clustering(comps, labeled, n_clusters):
    spectral = SpectralClustering(n_clusters=n_clusters).fit(comps)
    clusters = {}
    for x, label in enumerate(spectral.labels_):
        clusters[x + 1] = label
    labeled_clusters = np.vectorize(lambda c: clusters[c])(labeled)
    return labeled_clusters


def draw_clusters(comps, labeled, filename, n_clusters=5, method='kmeans', ignored=set()):
    if method == 'kmeans':
        labeled_clusters = bi_k_means_clustering(comps, labeled, n_clusters)
        r_l, g_l, b_l = create_color_maps(n_clusters)
    elif method == 'hierarchical':
        labeled_clusters, seeds = hierarchical_clustering(comps, labeled, n_clusters)
        r_l, g_l, b_l = create_color_maps2(seeds)
    elif method == 'spectral':
        labeled_clusters = spectral_clustering(comps, labeled, n_clusters)
        r_l, g_l, b_l = create_color_maps(n_clusters)
    else:
        raise ValueError(f'Unknown method name: {method}. Possible values are [\'kmeans\', \'hierarchical\']')

    img_r = np.vectorize(r_l)(labeled_clusters)
    img_g = np.vectorize(g_l)(labeled_clusters)
    img_b = np.vectorize(b_l)(labeled_clusters)
    img = np.stack((img_r, img_g, img_b), axis=2).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def draw_labels(labeled, n_labels, filename):
    r_l, g_l, b_l = create_color_maps(n_labels)
    img_r = np.vectorize(r_l)(labeled - 1)
    img_g = np.vectorize(g_l)(labeled - 1)
    img_b = np.vectorize(b_l)(labeled - 1)
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
