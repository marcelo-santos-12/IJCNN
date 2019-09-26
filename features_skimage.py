import numpy as np
import collections
import cv2
from lbp_module.texture import local_binary_pattern
from skimage.feature import local_binary_pattern as lbp


def main():
    name_img = 'DATASET/img_sample/sample_lympho.tif'
    #name_img = '/home/marcelo/Imagens/Captura de tela de 2019-09-03 10-32-57.png'
    img = cv2.imread(name_img, 0)
    feature_skimage = lbp(img, P=8, R=1, method='default')
    feature_my = local_binary_pattern(img, P=8, R=1, method='default')
    #print('Original:\n ', np.unique(img))
    print('Module:\n ', np.unique(feature_my[0] == feature_skimage))
    #print('Skimage:\n ', np.unique(feature_skimage))

if __name__ == '__main__':

	main()