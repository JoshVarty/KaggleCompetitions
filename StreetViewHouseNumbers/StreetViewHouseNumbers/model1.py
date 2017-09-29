import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import os
from digit_struct import DigitStruct
from image_helpers import prep_data

if sys.platform == 'win32':
    os.chdir("C:\\git\\KaggleCompetitions\\StreetViewHouseNumbers\\StreetViewHouseNumbers")

image_size = 96
num_channels = 3
pixel_depth = 255.0  # Number of levels per pixel.
num_labels = 2
patch_size_3 = 3
depth = 32

TRAIN_DIR = "../input/train/"
digit_struct = DigitStruct(TRAIN_DIR + "/digitStruct.mat")
labels, paths = digit_struct.load_labels_or_extract("../input/train/labels_and_paths.pickle")

image_paths = [TRAIN_DIR + s for s in paths]

train_normalized = prep_data(image_paths, image_size, num_channels, pixel_depth)
print("Train shape: {}".format(train_normalized.shape))

#x = labels[2]
#plt.imshow(train_normalized[2])
#plt.show()


#x = labels[4]
#plt.imshow(train_normalized[4])
#plt.show()

#np.random.seed(42)
#def randomize(dataset, labels):
#  permutation = np.random.permutation(labels.shape[0])
#  shuffled_dataset = dataset[permutation,:,:,:]
#  shuffled_labels = labels[permutation]
#  return shuffled_dataset, shuffled_labels



