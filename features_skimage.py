import numpy as np
import collections
import cv2
from lbp_module.texture import local_binary_pattern
from skimage.feature import local_binary_pattern as lbp


def main():
    name_img = 'DATASET/img_sample/sample_lympho.tif'
    #name_img = '/home/marcelo/Imagens/Captura de tela de 2019-09-03 10-32-57.png'
    img = cv2.imread(name_img, 0)
    feature_skimage = lbp(img, P=8, R=1, method='uniform')
    feature_my = local_binary_pattern(img, P=8, R=1, method='uniform')
    print('Original:\n ', np.unique(img))
    print('Module:\n ', np.unique(feature_my))
    print('Skimage:\n ', np.unique(feature_skimage))

    teste = feature_my != feature_skimage
    print(teste)
    a = []
    b = []
    for i in range(teste.shape[0]):
        for j in  range(teste.shape[1]):
            if teste[i, j]:
                a.append(feature_my[i, j])
                b.append(feature_skimage[i, j])
    
    a = np.array(a)
    unique, counts = np.unique(a, return_counts=True)
    RESULTS = {}
    for i in range(unique.shape[0]):
        RESULTS[str(unique[i])] = counts[i]
    print(RESULTS)

    b = np.array(b)
    unique, counts = np.unique(b, return_counts=True)
    RESULTS = {}
    for i in range(unique.shape[0]):
        RESULTS[str(unique[i])] = counts[i]
    print(RESULTS)

    np.savetxt('teste.txt', np.array([a.T, b.T]))

if __name__ == '__main__':

	main()