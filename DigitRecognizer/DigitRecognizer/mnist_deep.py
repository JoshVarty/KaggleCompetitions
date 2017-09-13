import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from random import randint
import numpy as np

image_size = 28     #28x28 images
num_channels = 1    #Not RGB, just single values for each pixel
num_labels = 10     #0-9

batch_size = 128
patch_size = 4
depth = 16
num_hidden = 64


#Load data  
data = pd.read_csv("../input/train.csv");
test_data = pd.read_csv("../input/test.csv");
test_data = test_data.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)


#Partition into train/test sets
images = data.iloc[0:5000,1:]
labels = data.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#Cleanup
del data
del images
del labels


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  labels = np.reshape(labels, (len(labels), num_labels))
  return dataset, labels

print("Shapes Before Reformat")
print("train_images: ", train_images.shape)
print("train_labels: ", train_labels.shape)
print("test_images: ", test_images.shape)
print("test_labels: ", test_labels.shape)

#Convert to numpy arrays for tensorflow
train_images, train_labels = reformat(train_images.as_matrix(), train_labels.as_matrix())
test_images, test_labels = reformat(test_images.as_matrix(), test_labels.as_matrix())

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def ConvNet():

  graph = tf.Graph()
  with graph.as_default():
      def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

      def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

      def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

      
      def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

       # Input data.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_test_dataset = tf.constant(test_images, dtype=tf.float32)

      def model(data, keep_prob):
          # First convolutional layer - maps one grayscale image to 32 feature maps.
          with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)

          # Pooling layer - downsamples by 2X.
          with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

          # Second convolutional layer -- maps 32 feature maps to 64.
          with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

          # Second pooling layer.
          with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

          # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
          # is down to 7x7x64 feature maps -- maps this to 1024 features.
          with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

          # Dropout - controls the complexity of the model, prevents co-adaptation of features.
          with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

          # Map the 1024 features to 10 classes, one for each digit
          with tf.name_scope('fc2'):
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

          return y_conv

      
      # Training computation.
      logits = model(tf_train_dataset, 0.75)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        
      tf.summary.scalar("Loss", loss)

      # Optimizer.
      optimizer = tf.train.AdamOptimizer(0.0000005).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))

      num_steps = 20001

  with tf.Session(graph=graph) as session:
      merged = tf.summary.merge_all()
      writer = tf.summary.FileWriter('./train', session.graph)

      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

        if (step % 250 == 0):
          _, l, predictions, m = session.run([optimizer, loss, train_prediction, merged], feed_dict=feed_dict)
          writer.add_summary(m, step)
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
        else:
          #Don't pass in merged dictionary for better performance
          _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)


      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
      
      #Save session
      saver = tf.train.Saver()
      save_path = saver.save(session, "model/model.ckpt")

ConvNet();
