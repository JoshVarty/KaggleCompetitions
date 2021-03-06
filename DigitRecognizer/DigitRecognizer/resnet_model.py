import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from random import randint


image_size = 28
num_channels = 1
num_labels = 10

tensorboardPath = "/tmp/svhn_single"

train_data = pd.read_csv("../input/train.csv");
test_data = pd.read_csv("../input/test.csv");
test_data = test_data.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

train_labels = train_data.iloc[:,:1]
train_data = train_data.iloc[:,1:]

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset, labels

train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, train_size=0.90, random_state=0)

#Convert to numpy arrays for tensorflow
train_data, train_labels = reformat(train_data.as_matrix(), train_labels.as_matrix())
valid_data, valid_labels = reformat(valid_data.as_matrix(), valid_labels.as_matrix())

print("Train data", train_data.shape)
print("Train labels", train_labels.shape)
print("Valid data", valid_data.shape)
print("Valid labels", valid_labels.shape)
print("Test data", test_data.shape)

def TrainModel(min_lr, max_lr, stepsize, max_iter, name):

    print("MinLr", min_lr)
    print("MaxLr", max_lr)
    print("Stepsize", stepsize)
    print("MaxIter", max_iter)

    def bias_variable(name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

    def weight_layer(name, shape, initializer = None):
        if initializer is None:
            #Default to initializer for Relu layers
            initializer = tf.contrib.layers.variance_scaling_initializer()

        return tf.get_variable(name, shape, 
                               initializer=initializer,
                               regularizer=tf.contrib.layers.l2_regularizer(0.0001))

    def residual_block(net, num_channels, stage, block, is_training):
        weight_name_base = 'w' + str(stage) + block + '_branch'
        bias_name_base = 'b' + str(stage) + block + '_branch'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        shortcut = net

        #ConvLayer1
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      1
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2a', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2a', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2a') + bias
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2a', momentum=0.95, training=is_training)
        net = tf.nn.relu(net)

        #ConvLayer2
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      1
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2b', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2b', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2b') + bias
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2b', momentum=0.95, training=is_training)

        #Final step: Add shortcut value to main path
        net = tf.add(net, shortcut)
        net = tf.nn.relu(net)

        return net

    def downsample_block(net, num_channels, stage, block, stride, is_training):
        weight_name_base = 'w' + str(stage) + block + '_branch'
        bias_name_base = 'b' + str(stage) + block + '_branch'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        shortcut = net

        #ConvLayer1
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      2
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2a', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2a', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,2,2,1], padding='SAME', name=conv_name_base + '2a')
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2a', momentum=0.95, training=is_training)
        net = tf.nn.relu(net)

        #ConvLayer2
        #   padding     1   ("SAME") 
        #   kernel      3x3
        #   stride      1
        #   channels    num_channels
        shape = net.shape.as_list()
        weights = weight_layer(weight_name_base + '2b', [3, 3, shape[3], num_channels])
        bias = bias_variable(bias_name_base + '2b', [num_channels])
        net = tf.nn.conv2d(net, weights, strides=[1,1,1,1], padding='SAME', name=conv_name_base + '2b')
        net = tf.layers.batch_normalization(net, name=bn_name_base + '2b', momentum=0.95, training=is_training)

        #Avg Pool (of shortcut)
        #   padding     0   ("VALID") 
        #   kernel      3x3
        #   stride      2
        shortcut = tf.nn.avg_pool(shortcut, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        
        #Add shortcut to main path
        net = tf.add(shortcut, net)
        net = tf.nn.relu(net)

        #Concatenate with zeros
        zeros = tf.zeros_like(net)
        net = tf.concat([net, zeros], axis=3)

        return net
    
    def pad_and_random_crop(input):
        dynamic_shape = tf.shape(input)
        input = tf.image.resize_image_with_crop_or_pad(input, 32, 32)
        input = tf.random_crop(input, dynamic_shape)
        input = tf.reshape(input, [-1, image_size, image_size, num_channels])
        return input

    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels), name="input")
        labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        input = tf.cond(is_training,
                        lambda: pad_and_random_crop(input),
                        lambda: input)
        #If we're training, randomly flip the image
        #input = tf.cond(is_training,
        #                         lambda: random_flip_left_right(input),
        #                         lambda: input)

        #Stage 1
        shape = input.shape.as_list()
        weights = weight_layer("w_conv1", [3, 3, shape[3], 16])
        bias = bias_variable("b_conv1", [16])
        net = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='SAME') + bias
        net = tf.layers.batch_normalization(net, name="bn_conv1", momentum=0.95, training=is_training)
        net = tf.nn.relu(net)
        
        #Stage 2
        #RestNet Standard Block x9
        net = residual_block(net, num_channels=16, stage=2, block='a', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='b', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='c', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='d', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='e', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='f', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='g', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='h', is_training=is_training)
        net = residual_block(net, num_channels=16, stage=2, block='i', is_training=is_training)

        #Stage3
        #ResNet Downsample Block
        net = downsample_block(net, num_channels=16, stage=3, block='a', stride=2, is_training=is_training)
        #ResNet Standard Block x8
        net = residual_block(net, num_channels=32, stage=3, block='b', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='c', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='d', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='e', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='f', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='g', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='h', is_training=is_training)
        net = residual_block(net, num_channels=32, stage=3, block='i', is_training=is_training)

        #Stage4
        #ResNet Downsample Block
        net = downsample_block(net, num_channels=32, stage=4, block='a', stride=2, is_training=is_training)
        #ResNet Standard Block x8
        net = residual_block(net, num_channels=64, stage=4, block='b', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='c', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='d', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='e', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='f', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='g', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='h', is_training=is_training)
        net = residual_block(net, num_channels=64, stage=4, block='i', is_training=is_training)

        net = tf.nn.avg_pool(net, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
        shape = net.shape.as_list()
        reshape = tf.reshape(net, [-1, shape[3]])

        #User Xavier Initialization since we're going to run this through a SoftMax
        initializer = tf.contrib.layers.xavier_initializer()
        weight = weight_layer("w_fc", [shape[3], num_labels], initializer)
        bias = bias_variable("b_fc", [num_labels])
        logits = tf.matmul(reshape, weight) + bias

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        train_prediction = tf.nn.softmax(logits)
        
        correct_prediction = tf.equal(labels, tf.cast(tf.argmax(train_prediction, 1), tf.int32))
        tf_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope(name):
            tf.summary.scalar("loss", cost)
            tf.summary.scalar("accuracy", tf_accuracy)
            tf.summary.scalar("LR", learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        with tf.Session(graph=graph) as session:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(tensorboardPath)
            writer.add_graph(session.graph)

            batch_size = 125

            tf.global_variables_initializer().run()
            for step in range(max_iter + 1):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_data[offset:(offset + batch_size), :, :]
                batch_labels = np.squeeze(train_labels[offset:(offset + batch_size), :])

                cycle = np.floor(1.0 + step / (2.0 * stepsize))
                x = np.abs(float(step)/float(stepsize) - 2.0 * cycle + 1.0)
                lr = min_lr + (max_lr - min_lr) * np.max((0.0, 1.0 - x))

                feed_dict = {input : batch_data, labels : batch_labels, learning_rate: lr, is_training: True} 

                if step % 100 == 0:
                    _, l, predictions, m, acc = session.run([optimizer, cost, train_prediction, merged, tf_accuracy], feed_dict=feed_dict)
                    writer.add_summary(m, step)

                    if step % 500 == 0:
                        print('Minibatch loss at step %d: %f' % (step, l))
                        print('Minibatch accuracy: %.1f%%' % (acc * 100))
                else:
                    _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
                    
                    #If we ever end up getting NaNs, just end
                    if np.isnan(l):
                        print("Loss is NaN at step:", step)
                        break

                num_batches = 42
                if step % 500 == 0:
                    #See test set performance
                    accuracySum = 0.0

                    for i in range(0, len(valid_data), int(len(valid_data) / num_batches)):
                        batch_data = valid_data[i:i + int(len(valid_data) / num_batches)]
                        batch_labels = np.squeeze(valid_labels[i:i + int(len(valid_data) / num_batches)])
                        feed_dict = {input : batch_data, labels : batch_labels, learning_rate: lr, is_training: False} 
                        l, predictions, acc = session.run([cost, train_prediction, tf_accuracy], feed_dict=feed_dict)
                        accuracySum = accuracySum + acc

                    print('Test accuracy: %.1f%%' % ((accuracySum / num_batches) * 100))


            all_results = np.array([])
            #Now run the test
            #28,000 test images = 280 * 100
            num_steps = 280
            batch_size = 100
            for step in range(num_steps):
                offset = (step * batch_size) % (test_data.shape[0] - batch_size)
                batch_data = test_data[offset:(offset + batch_size), :, :, :]
                feed_dict = {input : batch_data, is_training : False}
                
                predictions = session.run([train_prediction], feed_dict=feed_dict)
                predictions = predictions[0]
                results = np.argmax(predictions, axis=1)
                results = np.squeeze(results)

                all_results = np.concatenate((all_results, results), axis=0)

            with open("results/results.csv", 'w') as file:
                file.write("ImageId,Label\n")
                for idx in range(len(all_results)):
                     prediction = int(all_results[idx])

                     file.write(str(idx + 1))
                     file.write(",")
                     file.write(str(prediction))
                     file.write("\n")



if __name__ == '__main__':
    try:
        shutil.rmtree(tensorboardPath)
    except:
        pass

    min_lr = 0.05
    max_lr = 0.5
    stepsize = 5000
    max_iter = 10000

    TrainModel(min_lr, max_lr, stepsize, max_iter, "Fig1b")
