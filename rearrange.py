# pytorch imports
import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.autograd import Variable

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import random
from tqdm import tqdm_notebook as tqdm
import math
import time
# Helper method to split dataset into train and test folders
def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print("\nCopying images into ",food)
        if not os.path.exists(os.path.join(dest,food)):
            os.makedirs(os.path.join(dest,food))
        for i in classes_images[food]:
            copy(os.path.join(src,food,i), os.path.join(dest,food,i))
    print("Copying Done!")

# # Prepare train dataset by copying images from food-101/images to food-101/train using the file train.txt
"""
print("Creating train data...")
META_PATH = dataset_path+"/food-101/meta/"
IMG_PATH = dataset_path+"/food-101/images/"
TRAIN_PATH = dataset_path+"/food-101-processed/train/"
prepare_data(META_PATH+'train.txt', IMG_PATH, TRAIN_PATH)
"""

# # Prepare validation data by copying images from food-101/images to food-101/valid using the file test.txt
"""
print("Creating validation data...")
META_PATH = dataset_path+"/food-101/meta/"
IMG_PATH = dataset_path+"/food-101/images/"
TEST_PATH = dataset_path+"/food-101-processed/test/"
prepare_data(META_PATH+'test.txt', IMG_PATH, TEST_PATH)
"""