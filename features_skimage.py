import numpy as np
import cv2
from lbp_module.texture import local_binary_pattern


def main():

	name_img = 'DATASET/img_sample/sample_lympho.tif'
	img = cv2.imread(name_img, 0)

	feature = local_binary_pattern(img, P=8, R=1)

	print(feature)

if __name__ == '__main__':

	main()
