import cv2
from src.segmenter.funcs import simplify_image, create_graph, draw_regions,method_3
from skimage.measure import label
from scipy.special import expit
import numpy as np
if __name__ == '__main__':
    do_simplification = False
    simplification_method = 'second_max'
    image_path = 'images/500.png'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if do_simplification:
        image = simplify_image(image, simplification_method)

    labeled, num_regions = label(image, return_num=True)
    print(f'Number of regions: {num_regions}')
    # label_classes, weights = create_graph(image, labeled)
    label_classes, weights = method_3(image, labeled, num_regions)
    print(weights.shape)
    np.save('adj.npy',weights)
    # d = 5
    # w = np.random.random((num_regions,d))
    # z = np.matmul(weights,w)
    # reconstructed = expit(np.matmul(z,z.T))
    # draw_image = draw_regions(image * 7, labeled)
    #
    # cv2.imwrite('out.png', draw_image)
