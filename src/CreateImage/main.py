import numpy as np
from scipy.spatial import distance_matrix

vectors = np.random.rand(*(10, 2))
a = np.array([[0, 0], [1, 1], [-1, -1]])


def get_medoid_index(arr):
    return np.argmin(np.sum(distance_matrix(arr, arr), axis=1))
