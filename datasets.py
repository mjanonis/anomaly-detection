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
    img1 = int(re.sub('\D', '', pair[0]))
    img2 = int(re.sub('\D', '', pair[1]))
    return int(img1 > img2)

# Returns the index of the side view image in a pair
def side_view(pair):
    img1 = int(re.sub('\D', '', pair[0]))
    img2 = int(re.sub('\D', '', pair[1]))
    return int(img1 < img2)


"""
Generates two .csv files from the root directory with the structure:

IMAGE, POSITIVE_PAIR
"""
def train_test_csv(root, train_size=0.8):
    # Get all the filepaths of the images
    filepaths = []
    pairs = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = subdir + os.sep + file
            filepaths.append(filepath)

    # Sort the filenames
    filepaths.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i in range(len(filepaths)):
        if i%2 == 0:
            pairs.append([filepaths[i], filepaths[i+1]])
        else:
            pairs.append([filepaths[i], filepaths[i-1]])

    shuffle(pairs)

    train_samples = int(train_size * len(pairs))

    # Generate the training .csv
    with open('train.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(pairs[:train_samples])

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

            if(v1_t == 0):
                pair[1] = pr[side_view(pr)]
            else:
                pair[1] = pr[top_view(pr)]

            pair.append(target)

    with open('test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test)

    

class SiameseXRayParcels(Dataset):
    def __init__(self, xray_csv, image_size=224, train=True, transform=False):
        self.train = train
        self.transform = transform
        self.pairs = []
        self.image_size = image_size
    
        # Read the .csv
        with open(xray_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.pairs.append(row)


    def __getitem__(self,index):

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
                if(v1_t == 0):
                    img2 = cv2.imread(pr[side_view(pr)])
                else:
                    img2 = cv2.imread(pr[top_view(pr)])

        else:
            img1 = cv2.imread(self.pairs[index][0])
            img2 = cv2.imread(self.pairs[index][1])
            target = int(self.pairs[index][2])

        img1, img2 = trim_xray_images(img1, img2, self.image_size+CROP_MARGIN, target)

        CROP_TOP_MAX_1 = img1.shape[0] - self.image_size
        CROP_TOP_MAX_2 = img2.shape[0] - self.image_size
        CROP_LEFT_MAX_1 = img1.shape[1] - self.image_size
        CROP_LEFT_MAX_2 = img2.shape[1] - self.image_size

        print(img1.shape, img2.shape)

        if self.train:
            if self.transform:
                # Apply random cropping
                top_1 = np.random.randint(0, CROP_TOP_MAX_1)
                top_2 = np.random.randint(0, CROP_TOP_MAX_2)
                left_1 = np.random.randint(0, CROP_LEFT_MAX_1)
                left_2 = np.random.randint(0, CROP_LEFT_MAX_2)

                img1 = img1[top_1:self.image_size+top_1, left_1:self.image_size+left_1]

                if target:
                    img2 = img2[top_2:self.image_size+top_2, left_1:self.image_size+left_1]
                else:
                    img2 = img2[top_2:self.image_size+top_2, left_2:self.image_size+left_2]

            else:
                # Apply a center crop
                top_1 = CROP_TOP_MAX_1 // 2
                top_2 = CROP_TOP_MAX_2 // 2
                left_1 = CROP_LEFT_MAX_1 // 2
                left_2 = CROP_LEFT_MAX_2 // 2

                img1 = img1[top_1:self.image_size+top_1, left_1:self.image_size+left_1]
                img2 = img2[top_2:self.image_size+top_2, left_2:self.image_size+left_2]

        else:
            # Apply a center crop
            top_1 = CROP_TOP_MAX_1 // 2
            top_2 = CROP_TOP_MAX_2 // 2
            left_1 = CROP_LEFT_MAX_1 // 2
            left_2 = CROP_LEFT_MAX_2 // 2

            img1 = img1[top_1:self.image_size+top_1, left_1:self.image_size+left_1]
            img2 = img2[top_2:self.image_size+top_2, left_2:self.image_size+left_2]


        # Normalize using the mean and std of ImageNet
        norm = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        
        #return (img1, img2), target

        # Convert the images to tensors and return
        return (norm(to_tensor(img1)), norm(to_tensor(img2))), target


    def __len__(self):
        return len(self.pairs)

