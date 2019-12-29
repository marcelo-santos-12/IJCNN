import numpy as np
import cv2
from lbp_module.texture import base_lbp, improved_lbp, hamming_lbp, completed_lbp, extended_lbp
import argparse
from skimage.feature import local_binary_pattern as lbp
import matplotlib.pyplot as plt
import time


def main():
    name_img = '/home/marcelo/Documentos/LGHM/IJCNN/DATASET/Breast Cancer/BreakHist_Dataset/40X/Benign/adenosis/SOB_B_A-14-22549AB-40-001.png'
    
    try:
        img = cv2.imread(name_img, 0)

    except Exception as e:
        raise ValueError(e)
    
    P, R = 8, 1
    a = time.time()
    feature_a = extended_lbp(img, P=P, R=R, method='default', block=(1,1))
    print('Tempo: ', time.time() - a)

    print(np.asarray(feature_a).shape)


if __name__ == '__main__':

	main()


'''

_bin = np.zeros(2**P)

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        px = output[i, j]
        _bin[int(px)] += 1
        
print(_bin)

'''