import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt

vecs = np.random.random((10, 5))
print(vecs)
# vecs = [[0, 0, 1], [1, 0, 0], [0, 0, 1.5], [2, 0, 0]]
# mat = distance_matrix(vecs, vecs)
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(vecs, 'ward',optimal_ordering=True)
print(linked)
labelList = range(0, len(vecs))

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',labels=labelList,
           distance_sort='descending',
           show_leaf_counts=True
           )
plt.show()
