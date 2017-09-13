import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from random import randint
import numpy as np

image_size = 28     #28x28 images
num_channels = 1    #Not RGB, just single values for each pixel
num_labels = 10     #0-9

patch_size_3 = 3
batch_size = 128

depth = 64

#Load data  
data = pd.read_csv("../input/train.csv");
test_data = pd.read_csv("../input/test.csv");
test_data = test_data.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)


#Partition into train/test sets
images = data.iloc[:,1:]
labels = data.iloc[:,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.99, random_state=0)

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


def TrainConvNet(model_save_path):

    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_test_dataset = tf.constant(test_images, dtype=tf.float32)
      
      layer1_weights = tf.get_variable("layer1_weights", [patch_size_3, patch_size_3, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
      layer1_biases = tf.get_variable("layer1_biases",[depth], initializer=tf.contrib.layers.xavier_initializer())
      layer2_weights = tf.get_variable("layer2_weights", [patch_size_3, patch_size_3, depth, depth], initializer=tf.contrib.layers.xavier_initializer())
      layer2_biases = tf.get_variable("layer2_biases",[depth], initializer=tf.contrib.layers.xavier_initializer())

      layer3_weights = tf.get_variable("layer3_weights", [patch_size_3, patch_size_3, depth, depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer3_biases = tf.get_variable("layer3_biases",[depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer4_weights = tf.get_variable("layer4_weights", [patch_size_3, patch_size_3, depth * 2, depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer4_biases = tf.get_variable("layer4_biases",[depth * 2], initializer=tf.contrib.layers.xavier_initializer())

      layer5_weights = tf.get_variable("layer5_weights", [patch_size_3, patch_size_3, depth * 2, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer5_biases = tf.get_variable("layer5_biases",[depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer6_weights = tf.get_variable("layer6_weights", [patch_size_3, patch_size_3, depth * 4, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer6_biases = tf.get_variable("layer6_biases", [depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer7_weights = tf.get_variable("layer7_weights", [patch_size_3, patch_size_3, depth * 4, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer7_biases = tf.get_variable("layer7_biases", [depth * 4], initializer=tf.contrib.layers.xavier_initializer())

      layer8_weights = tf.get_variable("layer8_weights", [patch_size_3, patch_size_3, depth * 4, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer8_biases = tf.get_variable("layer8_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer9_weights = tf.get_variable("layer9_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer9_biases = tf.get_variable("layer9_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer10_weights = tf.get_variable("layer10_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer10_biases = tf.get_variable("layer10_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      
      fc = 2048
      layer14_weights = tf.get_variable("layer14_weights", [fc, fc], initializer=tf.contrib.layers.xavier_initializer())
      layer14_biases = tf.get_variable("layer14_biases", [fc], initializer=tf.contrib.layers.xavier_initializer())

      layer15_weights = tf.get_variable("layer15_weights", [fc, num_labels], initializer=tf.contrib.layers.xavier_initializer())
      layer15_biases = tf.get_variable("layer15_biases", [num_labels], initializer=tf.contrib.layers.xavier_initializer())

      # Model
      def model(data, keep_prob):
        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer3_biases)
        conv = tf.nn.conv2d(hidden, layer4_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer4_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #Conv->Relu->Conv-Relu->Conv->Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer5_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer5_biases)
        conv = tf.nn.conv2d(hidden, layer6_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer6_biases)
        conv = tf.nn.conv2d(hidden, layer7_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer7_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #Conv->Relu->Conv-Relu->Conv->Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer8_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer8_biases)
        conv = tf.nn.conv2d(hidden, layer9_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer9_biases)
        conv = tf.nn.conv2d(hidden, layer10_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer10_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop = tf.nn.dropout(pool_1, keep_prob)
        shape = drop.get_shape().as_list()
        reshape = tf.reshape(drop, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.matmul(reshape, layer14_weights) + layer14_biases 
        drop = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(drop, layer15_weights) + layer15_biases 

      def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
      
      # Training computation.
      logits = model(tf_train_dataset, 0.5)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        
      tf.summary.scalar("Loss", loss)

      # Optimizer.
      optimizer = tf.train.AdamOptimizer(0.000001).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))

      num_steps = 30001

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
      save_path = saver.save(session, model_save_path)



def LoadAndRun(model_save_path):
    tf.reset_default_graph()
    graph = tf.Graph()
    num_steps = 280
    batch_size = int(len(test_data) / num_steps)

    with graph.as_default():
     
      # Input data.
      tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      
      layer1_weights = tf.get_variable("layer1_weights", [patch_size_3, patch_size_3, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
      layer1_biases = tf.get_variable("layer1_biases",[depth], initializer=tf.contrib.layers.xavier_initializer())
      layer2_weights = tf.get_variable("layer2_weights", [patch_size_3, patch_size_3, depth, depth], initializer=tf.contrib.layers.xavier_initializer())
      layer2_biases = tf.get_variable("layer2_biases",[depth], initializer=tf.contrib.layers.xavier_initializer())

      layer3_weights = tf.get_variable("layer3_weights", [patch_size_3, patch_size_3, depth, depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer3_biases = tf.get_variable("layer3_biases",[depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer4_weights = tf.get_variable("layer4_weights", [patch_size_3, patch_size_3, depth * 2, depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer4_biases = tf.get_variable("layer4_biases",[depth * 2], initializer=tf.contrib.layers.xavier_initializer())

      layer5_weights = tf.get_variable("layer5_weights", [patch_size_3, patch_size_3, depth * 2, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer5_biases = tf.get_variable("layer5_biases",[depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer6_weights = tf.get_variable("layer6_weights", [patch_size_3, patch_size_3, depth * 4, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer6_biases = tf.get_variable("layer6_biases", [depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer7_weights = tf.get_variable("layer7_weights", [patch_size_3, patch_size_3, depth * 4, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer7_biases = tf.get_variable("layer7_biases", [depth * 4], initializer=tf.contrib.layers.xavier_initializer())

      layer8_weights = tf.get_variable("layer8_weights", [patch_size_3, patch_size_3, depth * 4, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer8_biases = tf.get_variable("layer8_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer9_weights = tf.get_variable("layer9_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer9_biases = tf.get_variable("layer9_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer10_weights = tf.get_variable("layer10_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer10_biases = tf.get_variable("layer10_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      
      fc = 2048
      layer14_weights = tf.get_variable("layer14_weights", [fc, fc], initializer=tf.contrib.layers.xavier_initializer())
      layer14_biases = tf.get_variable("layer14_biases", [fc], initializer=tf.contrib.layers.xavier_initializer())

      layer15_weights = tf.get_variable("layer15_weights", [fc, num_labels], initializer=tf.contrib.layers.xavier_initializer())
      layer15_biases = tf.get_variable("layer15_biases", [num_labels], initializer=tf.contrib.layers.xavier_initializer())

      # Model
      def model(data, keep_prob):
        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer3_biases)
        conv = tf.nn.conv2d(hidden, layer4_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer4_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #Conv->Relu->Conv-Relu->Conv->Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer5_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer5_biases)
        conv = tf.nn.conv2d(hidden, layer6_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer6_biases)
        conv = tf.nn.conv2d(hidden, layer7_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer7_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #Conv->Relu->Conv-Relu->Conv->Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer8_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer8_biases)
        conv = tf.nn.conv2d(hidden, layer9_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer9_biases)
        conv = tf.nn.conv2d(hidden, layer10_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer10_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop = tf.nn.dropout(pool_1, keep_prob)
        shape = drop.get_shape().as_list()
        reshape = tf.reshape(drop, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.matmul(reshape, layer14_weights) + layer14_biases 
        drop = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(drop, layer15_weights) + layer15_biases 

      test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))


      results = np.array([])
      with tf.Session(graph=graph) as session:
         
          saver = tf.train.Saver()
          saver.restore(session, model_save_path)
          print("Restored")

          for step in range(num_steps):
            offset = (step * batch_size) % (test_data.shape[0] - batch_size)
            batch_data = test_data[offset:(offset + batch_size), :, :, :]
            feed_dict = {tf_test_dataset : batch_data}

            predictions  = session.run([test_prediction], feed_dict=feed_dict)
            predictions = predictions[0]
            batch_results = np.argmax(predictions, 1)

            results = np.concatenate((results, batch_results), axis=0)


      return results