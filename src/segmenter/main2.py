import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
import sys
import numpy
from collections import defaultdict, Counter
import time

draw_graph = False
verbose = False
if draw_graph:
    plt.rcParams["figure.figsize"] = (6, 6)
    # plt.autoscale(False)
    # plt.tight_layout()
if verbose:
    numpy.set_printoptions(threshold=sys.maxsize, linewidth=1000)

im = cv2.imread("500.png", cv2.IMREAD_GRAYSCALE)
nim = np.zeros(im.shape, dtype=im.dtype)
width, height = im.shape


def window(y, x, height,width,value):
    neighbors = []
    yl = y - 1 if y != 0 else y
    yu = y + 2 if y != height - 1 else y + 1
    xl = x - 1 if x != 0 else x
    xu = x + 2 if x != width - 1 else x + 1

    for yp in range(yl, yu):
        for xp in range(xl, xu):
            if yp != y or xp != x:
                tv = im[yp, xp]
                if value == tv:
                    return True, 0
                neighbors.append(tv)
    # c = Counter(neighbors).most_common(2)[-1][0]
    # c = max(neighbors,key=neighbors.count)
    c = min(neighbors,key=neighbors.count)
    return False, c


for x in range(width):
    for y in range(height):
        if y == 7 and x == 3:
            r = 1
        value = im[y, x]

        near, new_value = window(y, x, height,width,value)
        if near:
            nim[y, x] = value
        else:
            nim[y, x] = new_value

# cv2.imwrite('out.png', nim*15)
# cv2.imwrite('sc.png',im*15)
# changes = ((im-nim) > 0).astype('uint8') * 255

# cv2.imwrite('changes.png', changes)
labeled, num_labels = label(im, return_num=True, connectivity=2)
print(f'Num Regions: {num_labels}')
props = regionprops(labeled)
areas = [x.area for x in props]
# exit()
print(Counter(areas))
# bins = np.histogram_bin_edges(areas,bins=1000)
# plt.hist(areas,bins=bins)
# plt.show()
print(sum(areas) / num_labels)

if draw_graph:
    x, y = im.shape
    pos = {i: [value.centroid[1] / x, 1 - (value.centroid[0] / y)] for i, value in zip(range(1, num_labels + 1), props)}
    if verbose:
        print(pos)
        print(labeled)

width, height = labeled.shape


# weights = defaultdict(lambda: defaultdict(int))


def method_1(labeled):
    label_classes = {}
    weights = defaultdict(lambda: defaultdict(int))
    for x in range(width):
        for y in range(height):
            value = labeled[y, x]
            if value not in label_classes:
                label_classes[value] = im[y, x]
            if y != 0:
                weights[value][labeled[y - 1, x]] += 1
                # weights[value, labeled[y - 1, x]] += 1
            if x != 0:
                weights[value][labeled[y, x - 1]] += 1
                # weights[value, labeled[y, x - 1]] += 1
            if y != height - 1:
                weights[value][labeled[y + 1, x]] += 1
                # weights[value, labeled[y + 1, x]] += 1
            if x != width - 1:
                weights[value][labeled[y, x + 1]] += 1
                # weights[value, labeled[y, x + 1]] += 1
    return weights

weights = method_1(labeled)
v = len(weights)
e = 0
s = []


for key,item in weights.items():
    counter = 0
    for key2, con in item.items():
        if key != key2:
            counter+=1
            e +=.5
    s.append(counter)

print(v,e)
print(Counter(s))
bins = np.histogram_bin_edges(s,bins=300)
plt.hist(s,bins=bins)
plt.show()
print(sum(s)/len(s))
exit()


def method_2(labeled):
    label_classes = {}
    weights = sparse.lil_matrix((num_labels + 1, num_labels + 1), dtype='int32')
    for x in range(width):
        for y in range(height):
            value = labeled[y, x]
            if value not in label_classes:
                label_classes[value] = im[y, x]
            if y != 0:
                weights[value, labeled[y - 1, x]] += 1
            if x != 0:
                weights[value, labeled[y, x - 1]] += 1
            if y != height - 1:
                weights[value, labeled[y + 1, x]] += 1
            if x != width - 1:
                weights[value, labeled[y, x + 1]] += 1
    return weights.nnz * 4


def method_3(labeled):
    label_classes = {}
    weights = np.zeros((num_labels + 1, num_labels + 1), dtype='int32')
    for x in range(width):
        for y in range(height):
            value = labeled[y, x]
            if value not in label_classes:
                label_classes[value] = im[y, x]
            if y != 0:
                weights[value, labeled[y - 1, x]] += 1
            if x != 0:
                weights[value, labeled[y, x - 1]] += 1
            if y != height - 1:
                weights[value, labeled[y + 1, x]] += 1
            if x != width - 1:
                weights[value, labeled[y, x + 1]] += 1
    return weights.nbytes


import random

methods = [(1, method_1), (3, method_3)]

tots = [0, 0, 0]

for _ in range(10):
    for id, m in methods:
        s = time.perf_counter()
        v = m(labeled)
        t = time.perf_counter() - s
        tots[id - 1] += t
        print(f'id: {id}, Time: {t}, Size: {v}')
    random.shuffle(methods)

print(tots)
# weights = weights[1:, 1:]
# print(weights)

print('Counting')
# print(f'Number of edges: {np.count_nonzero(weights)}')
# if verbose:
#     print(weights)
#     print(label_classes)

if draw_graph:
    G = nx.from_numpy_matrix(weights)
    relabeld = {x: x + 1 for x in range(num_labels)}
    G = nx.relabel_nodes(G, relabeld)
    # pos = nx.spring_layout(G)
    # pos[1] = [0,1]
    print(pos)
    labels = nx.get_edge_attributes(G, 'weight')
    # print(labels)

    # nx.draw(G, pos, with_labels=True)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
