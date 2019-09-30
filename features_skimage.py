import numpy as np
import cv2
from lbp_module.texture import local_binary_pattern
import argparse
from skimage.feature import local_binary_pattern as lbp


def main():
    
    name_img = 'DATASET/Breast Cancer/breakhist-dataset/BreakHist_Dataset/40X/Benign/adenosis/SOB_B_A-14-22549AB-40-001.png'
    try:
        img = cv2.imread(name_img, 0)

    except Exception as e:
        raise Exception(e)

    feature = local_binary_pattern(img, P=P, R=R)

    #cv2.imshow('test', feature)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
	
    feature_skimage = lbp(img, P=8, R=1)
	feature_my = local_binary_pattern(img, P=8, R=1)
	
    print('Original:\n ', img.shape)
	print('Module:\n ', feature_my.shape)
	print('Skimage:\n ', feature_skimage.shape)

if __name__ == '__main__':

	main()
