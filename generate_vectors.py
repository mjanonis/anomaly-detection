# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2020, Martynas Janonis

# Licensed under the EUPL-1.2-or-later

import os
import re
import numpy as np
import cv2
import math
import torch

import trim

from skimage.util import view_as_windows
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor


def generate_vectors(root, dest, model):
    model.eval()

    filepaths = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            filepaths.append(filepath)

    # Sort the filenames
    filepaths.sort(key=lambda f: int(re.sub("\D", "", f)))

    for file in filepaths:
        print(file)
        patches = produce_patches(file)
        print(len(patches))
        f = open(dest + os.path.splitext(os.path.basename(file))[0] + ".vec", "ab")
        for patch in patches:
            with torch.no_grad():
                vec = model(patch).cpu().detach().numpy()
            np.savetxt(f, vec, delimiter=",")
        f.close()


def produce_patches(path):
    img = cv2.imread(path)
    patches = []
    # Trim the image
    img = trim.trim_xray_images(img, img, 224, 0)[0]

    # Add a white border around the image to make sure that stride of 112 covers it entirely
    img = cv2.copyMakeBorder(
        img,
        (round_to(img.shape[0], 112) - img.shape[0]) // 2,
        (round_to(img.shape[0], 112) - img.shape[0]) // 2,
        (round_to(img.shape[1], 112) - img.shape[1]) // 2,
        (round_to(img.shape[1], 112) - img.shape[1]) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )

    window = view_as_windows(img, (224, 224, 3))

    # Normalize using the mean and std of ImageNet
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for h in range(0, window.shape[0], 112):
        for w in range(0, window.shape[1], 112):
            patches.append(norm(to_tensor(window[h][w][0])).unsqueeze(0))

    return patches
