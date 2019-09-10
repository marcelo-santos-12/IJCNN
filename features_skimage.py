import numpy as np
import cv2
from lbp_module.texture import local_binary_pattern
from skimage.feature import local_binary_pattern as lbp


def main():
    name_img = 'DATASET/img_sample/sample_lympho.tif'
    img = cv2.imread(name_img, 0)
    feature_skimage = lbp(img, P=4, R=1)
    feature_my = local_binary_pattern(img, P=4, R=1)
    print('Original:\n ', img.shape)
    print('Module:\n ', feature_my.shape)
    print('Skimage:\n ', feature_skimage.shape)

if __name__ == '__main__':

	main()
