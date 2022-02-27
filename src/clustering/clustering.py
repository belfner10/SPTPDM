import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage
from skimage.measure import label

from src.clustering.funcs import create_clusters
from src.segmenter.funcs import simplify_image


def create_several_clustes(vecs, image_path):
    Z = linkage(vecs, 'complete')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


if __name__ == '__main__':
    vecs = np.load('out_grarep.np.npy')[:, 1:]

    Z = linkage(vecs, 'complete')

    image = cv2.imread('1000.png', cv2.IMREAD_GRAYSCALE)
    image = simplify_image(image, 'second_max')
    labeled, num_regions = label(image, return_num=True)
    print('Creating Images')
    for r in range(2, 10):
        print(f'Creating Image {r}')
        m = create_clusters(Z, len(vecs), r)
        mult = int(250 / r)
        out = np.zeros(labeled.shape, dtype='uint8')
        for y in range(labeled.shape[0]):
            for x in range(labeled.shape[1]):
                out[y, x] = m[labeled[y, x]] * mult
        cv2.imwrite(f'out{r}.png', out)