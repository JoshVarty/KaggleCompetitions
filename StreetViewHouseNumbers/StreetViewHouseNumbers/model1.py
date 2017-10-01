import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
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
num_labels = 5
patch_size_3 = 3
depth = 32

TRAIN_DIR = "../input/train/"
digit_struct = DigitStruct(TRAIN_DIR + "/digitStruct.mat")
labels, paths = digit_struct.load_labels_or_extract("../input/train/labels_and_paths.pickle")


labels = np.array(labels)

image_paths = [TRAIN_DIR + s for s in paths]

train_normalized = prep_data(image_paths, image_size, num_channels, pixel_depth)
print("Train shape: {}".format(train_normalized.shape))

np.random.seed(42)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def accuracy(predictions, labels):
    labels1 = labels[:,0,:]
    labels2 = labels[:,1,:]
    labels3 = labels[:,2,:]
    labels4 = labels[:,3,:]
    labels5 = labels[:,4,:]

    pred1 = predictions[:,0,:]
    pred2 = predictions[:,1,:]
    pred3 = predictions[:,2,:]
    pred4 = predictions[:,3,:]
    pred5 = predictions[:,4,:]

    num1 = np.sum(np.argmax(pred1, 1) == np.argmax(labels1,1))
    num2 = np.sum(np.argmax(pred2, 1) == np.argmax(labels2,1))
    num3 = np.sum(np.argmax(pred3, 1) == np.argmax(labels3,1))
    num4 = np.sum(np.argmax(pred4, 1) == np.argmax(labels4,1))
    num5 = np.sum(np.argmax(pred5, 1) == np.argmax(labels5,1))

    return 100 * (num1 + num2 + num3 + num4 + num5) / (predictions.shape[0] * predictions.shape[1])

train_dataset_rand, train_labels_rand = randomize(train_normalized, labels)
train_images, valid_images, train_labels, valid_labels = train_test_split(train_dataset_rand, train_labels_rand, train_size=0.9, random_state=0)



def TrainConvNet(model_save_path):

    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
        labels = tf.placeholder(tf.float32, shape=(None, 5, 11))
        #keep_prob = tf.placeholder(tf.float32)

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
        h_conv5 = conv2d_relu(h_pool4, w_conv5, b_conv5)
        w_conv6 = weight_layer("w_conv6", [patch_size_3, patch_size_3, depth * 4, depth * 4])
        b_conv6 = bias_variable("b_conv6", [depth * 4])
        h_conv6 = conv2d_relu(h_conv5, w_conv6, b_conv6)
        w_conv7 = weight_layer("w_conv7", [patch_size_3, patch_size_3, depth * 4, depth * 8])
        b_conv7 = bias_variable("b_conv7", [depth * 8])
        h_pool7 = max_pool_2x2(h_conv6)

        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop_1 = tf.nn.dropout(h_pool7, 1.0)
        shape = drop_1.get_shape().as_list()
        reshape = tf.reshape(drop_1, [-1, shape[1] * shape[2] * shape[3]])

        fc = 18432
        w_fc_1 = weight_layer("w_fc_1", [fc, 4096])
        b_fc_1 = bias_variable("b_fc_1", [4096])
        h_fc_1 = tf.matmul(reshape, w_fc_1) + b_fc_1

        drop_2 = tf.nn.dropout(h_fc_1, 1.0)
        w_fc_2 = weight_layer("w_fc_2", [4096, 11 * num_labels])
        b_fc_2 = bias_variable("b_fc_2", [11 * num_labels])
        h_fc_2 = tf.matmul(drop_2, w_fc_2) + b_fc_2


        labels1 = tf.squeeze(tf.slice(labels, [0, 0, 0], [-1, 1, 11]), axis=1)
        labels2 = tf.squeeze(tf.slice(labels, [0, 1, 0], [-1, 1, 11]), axis=1)
        labels3 = tf.squeeze(tf.slice(labels, [0, 2, 0], [-1, 1, 11]), axis=1)
        labels4 = tf.squeeze(tf.slice(labels, [0, 3, 0], [-1, 1, 11]), axis=1)
        labels5 = tf.squeeze(tf.slice(labels, [0, 4, 0], [-1, 1, 11]), axis=1)

        logits1 = tf.slice(h_fc_2, [0, 0], [-1, 11])
        logits2 = tf.slice(h_fc_2, [0, 11], [-1, 11])
        logits3 = tf.slice(h_fc_2, [0, 22], [-1, 11])
        logits4 = tf.slice(h_fc_2, [0, 33], [-1, 11])
        logits5 = tf.slice(h_fc_2, [0, 44], [-1, 11])

        cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels1, logits=logits1))
        cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels2, logits=logits2))
        cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels3, logits=logits3))
        cost4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels4, logits=logits4))
        cost5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels5, logits=logits5))

        total_cost = cost1 + cost2 + cost3 + cost4 + cost5

        train_prediction = tf.stack([
            tf.nn.softmax(logits1),
            tf.nn.softmax(logits2),
            tf.nn.softmax(logits3),
            tf.nn.softmax(logits4),
            tf.nn.softmax(logits5),
            ], axis=1)

        optimizer = tf.train.AdamOptimizer(0.00001).minimize(total_cost)

        #Slice it out
        #x = tf.slice(h_fc_2, [0,0,0], [-1, -1, - 1])

        with tf.Session(graph=graph) as session:
            num_steps = 10000
            batch_size = 32
            tf.global_variables_initializer().run()
            print("Initialized")

            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_images[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]

                feed_dict = {input : batch_data, labels : batch_labels}

                if step % 500 == 0:
                    _, l, predictions = session.run([optimizer, total_cost, train_prediction], feed_dict=feed_dict)
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    #Validation

                    v_steps = 5
                    v_batch_size = int(valid_images.shape[0] / v_steps)
                    v_preds = np.zeros_like(valid_labels)
                    for v_step in range(v_steps):
                        v_offset = (v_step * v_batch_size) 
                        v_batch_data = valid_images[v_offset:(v_offset + v_batch_size), :, :, :]
                        v_batch_labels = valid_labels[v_offset:(v_offset + v_batch_size),:]

                        feed_dict = {input : v_batch_data, labels : v_batch_labels}
                        _, l, predictions = session.run([optimizer, total_cost, train_prediction], feed_dict=feed_dict)
                        v_preds[v_offset: v_offset + predictions.shape[0],:,:] = predictions

                    print('Valid accuracy: %.1f%%' % accuracy(v_preds, valid_labels))
                else:
                    _, l, predictions = session.run([optimizer, total_cost, train_prediction], feed_dict=feed_dict)

def weight_layer(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(name, shape):
      return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d_relu(input, weights, bias):
    return tf.nn.relu(tf.nn.conv2d(input, weights, [1,1,1,1], padding="SAME") + bias)

def max_pool_2x2(input):    
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


TrainConvNet("")



    



#x = labels[2]
#plt.imshow(train_normalized[2])
#plt.show()


#x = labels[4]
#plt.imshow(train_normalized[4])
#plt.show()



