# SPDX-License-Identifier: EUPL-1.2

# Unmodified code written by Adam Bielski is licensed under the BSD-3-Clause license

# All further additions and modifications: Copyright (c) 2020, Martynas Janonis

# Licensed under the EUPL-1.2-or-later

import numpy as np
import cv2
import os
import re
import csv

from PIL import Image
from random import shuffle, choice

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize

from trim import trim_xray_images

CROP_MARGIN = 32

# Returns the index of the top view image in a pair
def top_view(pair):
    img1 = int(re.sub("\D", "", pair[0]))
    img2 = int(re.sub("\D", "", pair[1]))
    return int(img1 > img2)


# Returns the index of the side view image in a pair
def side_view(pair):
    img1 = int(re.sub("\D", "", pair[0]))
    img2 = int(re.sub("\D", "", pair[1]))
    return int(img1 < img2)


"""
Generates two .csv files from the root directory with the structure:

IMAGE, POSITIVE_PAIR
"""


def siamese_train_test_csv(root, train_size=0.8):
    # Get all the filepaths of the images
    filepaths = []
    pairs = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            filepaths.append(filepath)

    # Sort the filenames
    filepaths.sort(key=lambda f: int(re.sub("\D", "", f)))

    for i in range(len(filepaths)):
        if i % 2 == 0:
            pairs.append([filepaths[i], filepaths[i + 1]])
        else:
            pairs.append([filepaths[i], filepaths[i - 1]])

    train_samples = int(train_size * len(pairs))

    # Generate the training .csv
    with open("train.csv", "w") as csvfile:
        train = pairs[:train_samples].copy()
        shuffle(train)
        writer = csv.writer(csvfile)
        writer.writerows(train)

    # Generate the testing .csv

    test = pairs[train_samples:]
    for pair in test:
        target = np.random.randint(0, 2)
        if target:
            pair.append(target)
        else:
            pr = pair
            while pr == pair or pr[0] == pair[1]:
                pr = choice(test)
            v1_t = top_view(pair)

            if v1_t == 0:
                pair[1] = pr[side_view(pr)]
            else:
                pair[1] = pr[top_view(pr)]

            pair.append(target)

    with open("test.csv", "w") as csvfile:
        shuffle(test)
        writer = csv.writer(csvfile)
        writer.writerows(test)


# Outputs all pairs to a .csv file
def triplet_train_test_csv(root, train_size=0.8):
    # Get all the filepaths of the images
    filepaths = []
    pairs = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            filepaths.append(filepath)

    # Sort the filenames
    filepaths.sort(key=lambda f: int(re.sub("\D", "", f)))

    for i in range(len(filepaths)):
        if i % 2 == 0:
            pairs.append([filepaths[i], filepaths[i + 1]])

    train_samples = int(train_size * len(pairs))
    shuffle(pairs)

    # Generate the training .csv
    with open("triplet_train.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(pairs[:train_samples])

    # Generate anchor, positive, negative triplets
    test = pairs[train_samples:]
    test_set = []
    for idx in range(0, (len(pairs) - train_samples) * 2):
        anchor = test[idx // 2][idx % 2]
        positive = test[idx // 2][(idx + 1) % 2]
        n_idx = idx
        while n_idx == idx:
            n_idx = np.random.randint(0, (len(pairs) - train_samples) * 2)
        negative = test[n_idx // 2][(idx + 1) % 2]
        test_set.append((anchor, positive, negative))

    with open("triplet_test.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test_set)


def svm_train_test_csv(root_neg, root_pos, train_size=0.8):
    # Get all the filepaths of the images
    filepaths = []
    for subdir, dirs, files in os.walk(root_neg):
        for file in files:
            filepath = subdir + os.sep + file
            filepaths.append((filepath, 0))

    for subdir, dirs, files in os.walk(root_pos):
        for file in files:
            filepath = subdir + os.sep + file
            filepaths.append((filepath, 1))

    shuffle(filepaths)
    train_samples = int(train_size * len(filepaths))

    with open("svm_train.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filepaths[:train_samples])

    with open("svm_test.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filepaths[train_samples:])


class SiameseXRayParcels(Dataset):
    def __init__(self, xray_csv, image_size=224, train=True, transform=False):
        self.train = train
        self.transform = transform
        self.pairs = []
        self.image_size = image_size

        # Read the .csv
        with open(xray_csv, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.pairs.append(row)

    def __getitem__(self, index):

        # Final images have to be at least 224x224

        if self.train:
            target = np.random.randint(0, 2)
            img1 = cv2.imread(self.pairs[index][0])
            if target:
                img2 = cv2.imread(self.pairs[index][1])
            else:
                pr = self.pairs[index]
                while pr == self.pairs[index] or pr[0] == self.pairs[index][1]:
                    pr = choice(self.pairs)
                v1_t = top_view(self.pairs[index])
                if v1_t == 0:
                    img2 = cv2.imread(pr[side_view(pr)])
                else:
                    img2 = cv2.imread(pr[top_view(pr)])

        else:
            img1 = cv2.imread(self.pairs[index][0])
            img2 = cv2.imread(self.pairs[index][1])
            target = int(self.pairs[index][2])

        img1, img2 = trim_xray_images(img1, img2, self.image_size + CROP_MARGIN, target)

        CROP_TOP_MAX_1 = img1.shape[0] - self.image_size
        CROP_TOP_MAX_2 = img2.shape[0] - self.image_size
        CROP_LEFT_MAX_1 = img1.shape[1] - self.image_size
        CROP_LEFT_MAX_2 = img2.shape[1] - self.image_size

        if self.train:
            if self.transform:
                # Apply random cropping
                top_1 = np.random.randint(0, CROP_TOP_MAX_1)
                top_2 = np.random.randint(0, CROP_TOP_MAX_2)
                left_1 = np.random.randint(0, CROP_LEFT_MAX_1)
                left_2 = np.random.randint(0, CROP_LEFT_MAX_2)

                img1 = img1[
                    top_1 : self.image_size + top_1, left_1 : self.image_size + left_1
                ]

                if target:
                    img2 = img2[
                        top_2 : self.image_size + top_2,
                        left_1 : self.image_size + left_1,
                    ]
                else:
                    img2 = img2[
                        top_2 : self.image_size + top_2,
                        left_2 : self.image_size + left_2,
                    ]

            else:
                # Apply a center crop
                top_1 = CROP_TOP_MAX_1 // 2
                top_2 = CROP_TOP_MAX_2 // 2
                left_1 = CROP_LEFT_MAX_1 // 2
                left_2 = CROP_LEFT_MAX_2 // 2

                img1 = img1[
                    top_1 : self.image_size + top_1, left_1 : self.image_size + left_1
                ]
                img2 = img2[
                    top_2 : self.image_size + top_2, left_2 : self.image_size + left_2
                ]

        else:
            # Apply a center crop
            top_1 = CROP_TOP_MAX_1 // 2
            top_2 = CROP_TOP_MAX_2 // 2
            left_1 = CROP_LEFT_MAX_1 // 2
            left_2 = CROP_LEFT_MAX_2 // 2

            img1 = img1[
                top_1 : self.image_size + top_1, left_1 : self.image_size + left_1
            ]
            img2 = img2[
                top_2 : self.image_size + top_2, left_2 : self.image_size + left_2
            ]

        # Normalize using the mean and std of ImageNet
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # return (img1, img2), target

        # Convert the images to tensors and return
        return (norm(to_tensor(img1)), norm(to_tensor(img2))), target

    def __len__(self):
        return len(self.pairs)


class TripletXRayParcels(Dataset):
    def __init__(self, pair_csv, image_size=224, train=True, transform=False):
        self.train = train
        self.transform = transform
        self.pairs = []
        self.image_size = image_size

        # Read the .csv
        with open(pair_csv, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.pairs.append(row)

    def __getitem__(self, index):
        # Prepare the triplet
        if self.train:
            anchor = cv2.imread(self.pairs[index // 2][index % 2])
            positive = cv2.imread(self.pairs[index // 2][(index + 1) % 2])
            n_ind = index
            while n_ind == index:
                n_ind = np.random.randint(0, len(self.pairs) * 2)
            negative = cv2.imread(self.pairs[n_ind // 2][(index + 1) % 2])
        else:
            anchor = cv2.imread(self.pairs[index][0])
            positive = cv2.imread(self.pairs[index][1])
            negative = cv2.imread(self.pairs[index][2])

        # Trim the images
        anchor, positive = trim_xray_images(
            anchor, positive, self.image_size + CROP_MARGIN, 1
        )
        _, negative = trim_xray_images(
            anchor, negative, self.image_size + CROP_MARGIN, 0
        )

        # Get maximum crop size
        CROP_TOP_MAX_A = anchor.shape[0] - self.image_size
        CROP_TOP_MAX_P = positive.shape[0] - self.image_size
        CROP_TOP_MAX_N = negative.shape[0] - self.image_size

        CROP_LEFT_MAX_A = anchor.shape[1] - self.image_size
        CROP_LEFT_MAX_P = positive.shape[1] - self.image_size
        CROP_LEFT_MAX_N = negative.shape[1] - self.image_size

        if self.train:
            if self.transform:
                # Apply random cropping
                top_a = np.random.randint(0, CROP_TOP_MAX_A)
                top_p = np.random.randint(0, CROP_TOP_MAX_P)
                top_n = np.random.randint(0, CROP_TOP_MAX_N)

                left_a = np.random.randint(0, CROP_LEFT_MAX_A)
                # Anchor and positive must be cropped from the same X coordinate
                left_p = left_a
                left_n = np.random.randint(0, CROP_LEFT_MAX_N)

                anchor = anchor[
                    top_a : self.image_size + top_a, left_a : self.image_size + left_a
                ]
                positive = positive[
                    top_p : self.image_size + top_p, left_p : self.image_size + left_p
                ]
                negative = negative[
                    top_n : self.image_size + top_n, left_n : self.image_size + left_n
                ]

            else:
                # Apply a center crop
                top_a = CROP_TOP_MAX_A // 2
                top_p = CROP_TOP_MAX_P // 2
                top_n = CROP_TOP_MAX_N // 2

                left_a = CROP_LEFT_MAX_A // 2
                left_p = CROP_LEFT_MAX_P // 2
                left_n = CROP_LEFT_MAX_N // 2

                anchor = anchor[
                    top_a : self.image_size + top_a, left_a : self.image_size + left_a
                ]
                positive = positive[
                    top_p : self.image_size + top_p, left_p : self.image_size + left_p
                ]
                negative = negative[
                    top_n : self.image_size + top_n, left_n : self.image_size + left_n
                ]

        else:
            # Apply a center crop
            top_a = CROP_TOP_MAX_A // 2
            top_p = CROP_TOP_MAX_P // 2
            top_n = CROP_TOP_MAX_N // 2

            left_a = CROP_LEFT_MAX_A // 2
            left_p = CROP_LEFT_MAX_P // 2
            left_n = CROP_LEFT_MAX_N // 2

            anchor = anchor[
                top_a : self.image_size + top_a, left_a : self.image_size + left_a
            ]
            positive = positive[
                top_p : self.image_size + top_p, left_p : self.image_size + left_p
            ]
            negative = negative[
                top_n : self.image_size + top_n, left_n : self.image_size + left_n
            ]

        # Normalize using the mean and std of ImageNet
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert the images to tensors and return
        return (
            (
                norm(to_tensor(anchor)),
                norm(to_tensor(positive)),
                norm(to_tensor(negative)),
            ),
            [],
        )

    def __len__(self):
        if self.train:
            return len(self.pairs * 2)
        else:
            return len(self.pairs)


class XRayParcels(Dataset):
    def __init__(self, csv_file, image_size=224, train=True, transform=False):
        self.train = train
        self.transform = transform
        self.data = []
        self.image_size = image_size

        # Read the .csv
        with open(csv_file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.data.append(row)

    def __getitem__(self, index):
        img = cv2.imread(self.data[index][0])
        label = np.int(self.data[index][1])

        img, _ = trim_xray_images(img, img, self.image_size + CROP_MARGIN, 0)

        CROP_TOP_MAX = img.shape[0] - self.image_size
        CROP_LEFT_MAX = img.shape[1] - self.image_size

        if self.train:
            if self.transform:
                # Apply random cropping
                top = np.random.randint(0, CROP_TOP_MAX)
                left = np.random.randint(0, CROP_LEFT_MAX)

                img = img[top : self.image_size + top, left : self.image_size + left]

            else:
                # Apply a center crop
                top = CROP_TOP_MAX // 2
                left = CROP_LEFT_MAX // 2

                img = img[top : self.image_size + top, left : self.image_size + left]
        else:
            # Apply a center crop
            top = CROP_TOP_MAX // 2
            left = CROP_LEFT_MAX // 2

            img = img[top : self.image_size + top, left : self.image_size + left]

        # Normalize using the mean and std of ImageNet
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return norm(to_tensor(img)), label

    def __len__(self):
        return len(self.data)
