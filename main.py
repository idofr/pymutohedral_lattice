from builtins import range
import numpy as np
from skimage.io import imread
from permutohedral_lattice import PermutohedralLattice
import logging
# from skimage.transform import resize
import sys
import cv2

__author__ = 'Ido Freeman'
__email__ = "idofreeman@gmail.com"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def main():
    # Read in image with the shape (rows, cols, channels)
    im = imread('./lena.small.jpg')
    im = np.array(im) / 255.

    invSpatialStdev = float(1. / 5.)
    invColorStdev = float(1. / .125)

    # Construct the position vectors out of x, y, r, g, and b.
    positions = np.zeros((im.shape[0], im.shape[1], 5), dtype='float32')
    for r in range(im.shape[0]):
        for c in range(im.shape[1]):
            positions[r, c, 0] = invSpatialStdev * c
            positions[r, c, 1] = invSpatialStdev * r
            positions[r, c, 2] = invColorStdev * im[r, c, 0]
            positions[r, c, 3] = invColorStdev * im[r, c, 1]
            positions[r, c, 4] = invColorStdev * im[r, c, 2]

    out = PermutohedralLattice.filter(im, positions)
    logging.info('Done')
    out -= out.min()
    out /= out.max()
    im -= im.min()
    im /= im.max()
    out = cv2.cvtColor(out.astype('float32'), cv2.COLOR_RGB2BGR)
    im = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2BGR)
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
    cv2.imshow('original', im)
    cv2.imshow('filtered', out)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
