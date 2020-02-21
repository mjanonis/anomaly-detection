# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2020, Martynas Janonis

# Licensed under the EUPL-1.2-or-later

import cv2
import numpy as np


def get_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

    # threshold to get just the signature
    retval, thresh_gray = cv2.threshold(
        gray, thresh=205, maxval=255, type=cv2.THRESH_BINARY
    )

    # find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray == 0)  # find where the black pixels are
    points = np.fliplr(
        points
    )  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points

    return x, y, w, h


def trim_xray_images(img1, img2, image_size, label):
    # add a white border around the images to make sure that there's enough space for cropping
    img1 = cv2.copyMakeBorder(
        img1,
        img2.shape[0] // 2 + image_size,
        img2.shape[0] // 2 + image_size,
        img2.shape[1] // 2 + image_size,
        img2.shape[1] // 2 + image_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    img2 = cv2.copyMakeBorder(
        img2,
        img1.shape[0] // 2 + image_size,
        img1.shape[0] // 2 + image_size,
        img1.shape[1] // 2 + image_size,
        img1.shape[1] // 2 + image_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )

    # get bounding box for both images
    x1, y1, w1, h1 = get_bounding_box(img1)
    x2, y2, w2, h2 = get_bounding_box(img2)

    # the bounding box has to be at least the size of the image_size
    if w1 < image_size:
        x1 -= (image_size - w1) // 2
        w1 += image_size - w1
    if h1 < image_size:
        y1 -= (image_size - h1) // 2
        h1 += image_size - h1

    if w2 < image_size:
        x2 -= (image_size - w2) // 2
        w2 += image_size - w2
    if h2 < image_size:
        y2 -= (image_size - h2) // 2
        h2 += image_size - h2

    # if two images are of the same parcel
    # make sure that the widths match
    if label:
        if w1 > w2:
            x2 -= (w1 - w2) // 2
            w2 = w1
        elif w2 > w1:
            x1 -= (w2 - w1) // 2
            w1 = w2

    # adjust the border if x or y is negative
    img1 = cv2.copyMakeBorder(
        img1,
        abs(min(0, y1)),
        0,
        abs(min(0, x1)),
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    img2 = cv2.copyMakeBorder(
        img2,
        abs(min(0, y2)),
        0,
        abs(min(0, x2)),
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    x2 = max(x2, 0)
    y2 = max(y2, 0)

    return img1[y1 : y1 + h1, x1 : x1 + w1], img2[y2 : y2 + h2, x2 : x2 + w2]

