from collections import Counter, defaultdict
from typing import Callable

import cv2
import numpy as np


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


def create_graph(image: np.ndarray, labeled_image: np.ndarray) -> (dict, defaultdict):
    """
    Creates adjacency matrix in the form of dictionary
    Parameters
    ----------
    image : np.ndarray
        image with class labels
    labeled_image : np.ndarray
        image with region labels
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
    return label_classes, weights


def draw_regions(image, regions: np.ndarray) -> np.ndarray:
    scale_factor = 10
    thickness = 0
    thickness = int(thickness / 2)
    heigth, width = image.shape
    drawing_image = cv2.resize(image, (heigth * scale_factor, width * scale_factor), interpolation=cv2.INTER_NEAREST)
    # cv2.rectangle(drawing_image,(0,100),(200,150),255,-1)
    for y in range(heigth):
        for x in range(width):
            if y != heigth - 1:
                if regions[y, x] != regions[y + 1, x]:
                    cv2.rectangle(drawing_image, (x * scale_factor - 1, (y + 1) * scale_factor - 1 - thickness), (x * scale_factor + scale_factor - 1, (y + 1) * scale_factor - 1 + thickness), 255, -1)
            if x != width - 1:
                if regions[y, x] != regions[y, x + 1]:
                    cv2.rectangle(drawing_image, ((x + 1) * scale_factor - 1 - thickness, y * scale_factor - 1), ((x + 1) * scale_factor - 1 + thickness, y * scale_factor + scale_factor - 1), 255, -1)
    return drawing_image


if __name__ == '__main__':
    simplify_image(1, 'test')
