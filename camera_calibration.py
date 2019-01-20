#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2 as cv
# built-in modules
import os

if __name__ == '__main__':
    import sys
    square_size =
    pattern_size = (7, 15)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    obj_points = []
    img_points = []

    img_names = os.listdir(r"C:\Users\NeilHazra\chessboard")

    h, w = cv.imread("C:\\Users\\NeilHazra\\chessboard\\" + img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]
    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread("C:\\Users\\NeilHazra\\chessboard\\" + fn, 0)
        img = cv.threshold(img, 155, 255, cv.THRESH_BINARY)[1]
        #cv.imshow("", img)
        #cv.waitKey(1)
        if img is None:
            print("Failed to load", fn)
            return None
        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img , pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if not found:
            print('chessboard not found')
            return None
        return (corners.reshape(-1, 2), pattern_points)
    threads_num = int(4)
    print("Run with %d threads..." % threads_num)
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(threads_num)
    chessboards = pool.map(processImage, img_names)
    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
