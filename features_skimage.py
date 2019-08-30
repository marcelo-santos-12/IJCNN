import numpy as np
import cv2
from lbp_module.texture import local_binary_pattern
import argparse



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True, \
            type=str, help="Path to the image")
    ap.add_argument("-p", "--points", required=False, default=8,\
	    type=int, help="Number of points surrounding each pixel")
    ap.add_argument("-r", "--radius", required=False, default=1, \
	    type=int, help="Radius of circle that surrounding each pixel")

    args = vars(ap.parse_args())

    name_img = args['input']
    try:
        img = cv2.imread(name_img, 0)

    except Exception as e:
        raise Exception(e)

    assert img.ndim == 2, 'Image do not have dimensions equal to 2'

    if args["points"]:
        P = args["points"]

    if args["radius"]:
        R = args["radius"]

    feature = local_binary_pattern(img, P=P, R=R)

    cv2.imshow('test', feature)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

	main()
