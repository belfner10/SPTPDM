from collections import Counter, defaultdict
from typing import Callable

import cv2
import numpy as np
import scipy.sparse as sp
from skimage.measure import label


def window(image: np.ndarray, y: int, x: int, value: int, method: Callable) -> (bool, int):
    """
    Analyzed the surrounding pixels to check if any adjacent pixels share the same class
    If none exist, choose a new pixel class based on the surrounding pixels
    Parameters
    ----------
    image: NDArray
        the passed image
    y: int
        y index of the pixel to be analyzed
    x: int
        x index of the pixel to be analyzed
    value: int
        class of the pixel to be analyzed TODO: this argument could be removed, time might suffer
    method: Callable
        method of selecting the new class of the pixel

    Returns
    -------

    """
    height, width = image.shape
    neighbors = []
    yl = y - 1 if y != 0 else y
    yu = y + 2 if y != height - 1 else y + 1
    xl = x - 1 if x != 0 else x
    xu = x + 2 if x != width - 1 else x + 1

    for yp in range(yl, yu):
        for xp in range(xl, xu):
            if yp != y or xp != x:
                tv = image[yp, xp]
                if value == tv:
                    return True, 0
                neighbors.append(tv)
    return False, method(neighbors)


def simplify_image(image: np.ndarray, method: str) -> np.ndarray:
    """

    Parameters
    ----------
    image : np.ndarray
        image to be processed
    method : str
        name of the method to use when selecting new pixel class

    Returns
    -------
    simplified_image : np.ndarray
        resulting simplified image
    """
    simplification_methods = {'max': lambda x: max(x, key=x.count),
                              'min': lambda x: min(x, key=x.count),
                              'second_max': lambda x: Counter(x).most_common(2)[-1][0]}
    if method not in simplification_methods.keys():
        possible_methods = "'" + "', '".join(simplification_methods.keys()) + "'"
        raise ValueError(f"'{method}' is not a valid simplification method. Available methods are: {possible_methods}")

    selected_method = simplification_methods[method]
    height, width = image.shape
    simplified_image = np.zeros(image.shape, dtype=image.dtype)

    # apply simplification to all pixel in image
    for x in range(width):
        for y in range(height):
            if y == 7 and x == 3:
                r = 1
            value = image[y, x]

            near, new_value = window(image, y, x, value, selected_method)
            if near:
                simplified_image[y, x] = value
            else:
                simplified_image[y, x] = new_value

    return simplified_image


def create_dict_adj(image: np.ndarray, labeled_image: np.ndarray, norm_weights: bool = False) -> (dict, defaultdict):
    """
    Creates adjacency matrix in the form of dictionary
    Parameters
    ----------
    image : np.ndarray
        image with class labels
    labeled_image : np.ndarray
        image with region labels
    norm_weights : bool
        flag that enables normalization of weights to the range 0-1
    Returns
    -------
    label_classes: dict
        dict mapping
    weights: defaultdict
        dictionary containing the connection weights between regions
    """
    height, width = labeled_image.shape
    label_classes = {}
    weights = defaultdict(lambda: defaultdict(int))
    for x in range(width):
        for y in range(height):
            region_label = labeled_image[y, x]
            if region_label not in label_classes:
                label_classes[region_label] = image[y, x]
            if y != 0:
                weights[region_label][labeled_image[y - 1, x]] += 1
            if x != 0:
                weights[region_label][labeled_image[y, x - 1]] += 1
            if y != height - 1:
                weights[region_label][labeled_image[y + 1, x]] += 1
            if x != width - 1:
                weights[region_label][labeled_image[y, x + 1]] += 1

    if not norm_weights:
        return label_classes, weights

    max_val = 0
    for key, value in weights.items():
        for v in value.values():
            max_val = max(max_val, v)

    for key, value in weights.items():
        for k2, v2 in value.items():
            if key < k2:
                value[k2] = v2 / max_val

    return label_classes, weights


def create_np_adj(image: np.ndarray, labeled_image: np.ndarray, num_labels: int, norm_weights: bool = False) -> (dict, np.array):
    """
    Creates adjacency matrix in the form of numpy array
    Parameters
    ----------
    image : np.ndarray
        image with class labels
    labeled_image : np.ndarray
        image with region labels
    num_labels: int
        number of regions found in the image
    norm_weights : bool
        flag that enables normalization of weights to the range 0-1
    Returns
    -------
    label_classes: dict
        dict mapping
    weights: np.array
        numpy adjacency matrix
    """
    label_classes = {}
    height, width = labeled_image.shape
    weights = np.zeros((num_labels + 1, num_labels + 1), dtype='float32')
    for x in range(width):
        for y in range(height):
            value = labeled_image[y, x]
            if value not in label_classes:
                label_classes[value] = image[y, x]
            if y != 0:
                weights[value, labeled_image[y - 1, x]] += 1
            if x != 0:
                weights[value, labeled_image[y, x - 1]] += 1
            if y != height - 1:
                weights[value, labeled_image[y + 1, x]] += 1
            if x != width - 1:
                weights[value, labeled_image[y, x + 1]] += 1

    for i in range(len(weights)):
        weights[i, i] = 0
    weights = weights[1:, 1:]
    if norm_weights:
        m = np.max(weights)
        weights /= m
    return label_classes, weights


def create_sp_adj(image: np.ndarray, labeled_image: np.ndarray, num_labels: int, norm_weights: bool = False) -> (dict, np.array):
    """
    Creates adjacency matrix in the form of numpy array
    Parameters
    ----------
    image : np.ndarray
        image with class labels
    labeled_image : np.ndarray
        image with region labels
    num_labels: int
        number of regions found in the image
    norm_weights : bool
        flag that enables normalization of weights to the range 0-1
    Returns
    -------
    label_classes: dict
        dict mapping
    weights: np.array
        numpy adjacency matrix
    """
    label_classes = {}
    height, width = labeled_image.shape
    weights = sp.lil_matrix((num_labels + 1, num_labels + 1), dtype='float32')
    for x in range(width):
        for y in range(height):
            value = labeled_image[y, x]
            if value not in label_classes:
                label_classes[value] = image[y, x]
            if y != 0:
                weights[value, labeled_image[y - 1, x]] += 1
            if x != 0:
                weights[value, labeled_image[y, x - 1]] += 1
            if y != height - 1:
                weights[value, labeled_image[y + 1, x]] += 1
            if x != width - 1:
                weights[value, labeled_image[y, x + 1]] += 1

    for i in range(weights.shape[0]):
        weights[i, i] = 0
    weights = weights[1:, 1:]
    weights = sp.coo_matrix(weights)
    weights.data = weights.data / np.max(weights.data)
    return label_classes, weights


def dict_to_sp(d, num_labels):
    weights = sp.lil_matrix((num_labels, num_labels), dtype='float32')
    for key, value in d.items():
        for k2, v2 in value.items():
            if key != k2:
                weights[key-1, k2-1] = v2
    return sp.csr_matrix(weights)


# unused
def draw_regions(image, regions: np.ndarray) -> np.ndarray:
    scale_factor = 10
    thickness = 0
    thickness = int(thickness / 2)
    heigth, width = image.shape
    drawing_image = cv2.resize(image, (heigth * scale_factor, width * scale_factor), interpolation=cv2.INTER_NEAREST)
    for y in range(heigth):
        for x in range(width):
            if y != heigth - 1:
                if regions[y, x] != regions[y + 1, x]:
                    cv2.rectangle(drawing_image, (x * scale_factor - 1, (y + 1) * scale_factor - 1 - thickness), (x * scale_factor + scale_factor - 1, (y + 1) * scale_factor - 1 + thickness), 255, -1)
            if x != width - 1:
                if regions[y, x] != regions[y, x + 1]:
                    cv2.rectangle(drawing_image, ((x + 1) * scale_factor - 1 - thickness, y * scale_factor - 1), ((x + 1) * scale_factor - 1 + thickness, y * scale_factor + scale_factor - 1), 255, -1)
    return drawing_image


def create_np_adj_from_image(image: np.array, smp_method: str = None, verbose=False):
    if smp_method:
        image = simplify_image(image, smp_method)
    labeled, num_regions = label(image, return_num=True)

    if verbose:
        print(f'Number of regions: {num_regions}')

    label_classes, weights = create_dict_adj(image, labeled, num_regions)
    weights = dict_to_sp(weights, num_regions)

    # label_classes, weights = create_sp_adj(image, labeled, num_regions)
    return label_classes, weights, labeled


if __name__ == '__main__':
    simplify_image(1, 'max')
