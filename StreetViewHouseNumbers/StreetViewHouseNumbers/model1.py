import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import os
from digit_struct import DigitStruct
from image_helpers import prep_data
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

np.random.seed(42)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset_rand, train_labels_rand = randomize(train_normalized, labels)
train_images, valid_images, train_labels, valid_labels = train_test_split(train_dataset_rand, train_labels_rand, train_size=0.8, random_state=0)



def TrainConvNet(model_save_path, keep_prob):
    
    input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
    keep_prob = tf.placeholder(tf.float32)

    #Conv->Relu->Conv->Relu->Pool
    w_conv1 = weight_layer("w_conv1", [patch_size_3, patch_size_3, num_channels, depth])
    b_conv1 = bias_variable("b_conv1", [depth])
    h_conv1 = conv2d_relu(input, w_conv1, b_conv1)
    w_conv2 = weight_layer("w_conv2", [patch_size_3, patch_size_3, depth, depth])
    b_conv2 = bias_variable("b_conv2", [depth])
    h_conv2 = conv2d_relu(h_conv1, w_conv2, b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #Conv->Relu->Conv->Relu->Pool
    w_conv3 = weight_layer("w_conv3", [patch_size_3, patch_size_3, depth, depth * 2])
    b_conv3 = bias_variable("b_conv3", [depth * 2])
    h_conv3 = conv2d_relu(h_pool2, w_conv3, b_conv3)
    w_conv4 = weight_layer("w_conv4", [patch_size_3, patch_size_3, depth * 2, depth * 4])
    b_conv4 = bias_variable("b_conv4", [depth * 4])
    h_conv3 = conv2d_relu(h_conv3, w_conv4, b_conv4)
    h_pool4 = max_pool_2x2(h_conv3)

    #Conv->Relu->Conv->Relu->Conv->Relu->Pool
    w_conv5 = weight_layer("w_conv5", [patch_size_3, patch_size_3, depth * 4, depth * 4])
    b_conv5 = bias_variable("b_conv5", [depth * 4])
    h_conv5 = conv2d_relu(h_pool4, w_conv4, b_conv4)
    w_conv6 = weight_layer("w_conv6", [patch_size_3, patch_size_3, depth * 4, depth * 4])
    b_conv6 = bias_variable("b_conv6", [depth * 4])
    h_conv6 = conv2d_relu(h_conv5, w_conv6, b_conv6)
    w_conv7 = weight_layer("w_conv7", [patch_size_3, patch_size_3, depth * 4, depth * 8])
    b_conv7 = bias_variable("b_conv7", [depth * 8])
    h_pool7 = max_pool_2x2(h_conv6)

    #Dropout -> Fully Connected -> Dropout -> Fully Connected
    drop_1 = tf.nn.dropout(h_pool7, keep_prob)
    shape = drop_1.get_shape().as_list()
    reshape = tf.reshape(drop_1, [shape[0], shape[1] * shape[2] * shape[3]])

    fc = 234
    w_fc_1 = weight_layer("w_fc_1", [fc, 4096])
    b_fc_1 = bias_variable("b_fc_1", [4096])
    hidden_1 = tf.matmul(reshape, w_fc_1) + b_fc_1

    drop_2 = tf.nn.dropout(hidden_1, keep_prob)
    w_fc_2 = weight_layer("w_fc_2", [4096, num_labels])
    b_fc_2 = bias_variable("b_fc_2", [num_labels])
    logits = tf.matmul(drop_2, w_fc_2) + b_fc_2

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)

def weight_layer(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(name, shape):
      return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d_relu(input, weights, bias):
    return tf.nn.relu(tf.nn.conv2d(input, weights, [1,1,1,1], padding="SAME") + bias)

def max_pool_2x2(input):    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')






    



#x = labels[2]
#plt.imshow(train_normalized[2])
#plt.show()


#x = labels[4]
#plt.imshow(train_normalized[4])
#plt.show()



