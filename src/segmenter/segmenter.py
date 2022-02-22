import cv2
from src.segmenter.funcs import simplify_image, create_graph, draw_regions,method_3
from skimage.measure import label
from scipy.special import expit
import numpy as np


if __name__ == '__main__':
    do_simplification = True
    simplification_method = 'second_max'
    image_path = 'images/1000.png'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite('gt.png',image*14)

    if do_simplification:
        image = simplify_image(image, simplification_method)
    cv2.imwrite('gts.png', image*14)
    labeled, num_regions = label(image, return_num=True)


    print(f'Number of regions: {num_regions}')
    # label_classes, weights = create_graph(image, labeled)
    labeled_classes, weights = create_graph(image,labeled,norm_weights=True)

    data = 'id_1,id_2,id_3\n'
    for key, value in weights.items():
        for k2, v2 in value.items():
            if key < k2:
                data +=f'{key-1},{k2-1},{v2}\n'

    with open('out.csv','w') as outfile:
        outfile.write(data)




    # label_classes, weights = method_3(image, labeled, num_regions)
    # print(weights.shape)
    # np.save('adj.npy',weights)

    # d = 5
    # w = np.random.random((num_regions,d))
    # z = np.matmul(weights,w)
    # reconstructed = expit(np.matmul(z,z.T))
    # draw_image = draw_regions(image * 7, labeled)
    #
    # cv2.imwrite('out.png', draw_image)
