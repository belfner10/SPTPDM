import cv2
from src.segmenter.funcs import simplify_image, create_graph, draw_regions
from skimage.measure import label

if __name__ == '__main__':
    do_simplification = False
    simplification_method = 'second_max'
    image_path = 'images/500.png'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if do_simplification:
        image = simplify_image(image, simplification_method)

    labeled, num_regions = label(image, return_num=True)

    label_classes, weights = create_graph(image, labeled)

    draw_image = draw_regions(image * 7, labeled)

    cv2.imwrite('out.png', draw_image)
