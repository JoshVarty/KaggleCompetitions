import os
import sys
import numpy as np
import tensorflow as tf
import cv2




def ConvNet(model_save_path):

    depth = 64
    batch_size = 16
    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
      
      #64
      layer1_weights = tf.get_variable("layer1_weights", [patch_size_3, patch_size_3, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
      layer1_biases = tf.get_variable("layer1_biases",[depth], initializer=tf.contrib.layers.xavier_initializer())
      layer2_weights = tf.get_variable("layer2_weights", [patch_size_3, patch_size_3, depth, depth], initializer=tf.contrib.layers.xavier_initializer())
      layer2_biases = tf.get_variable("layer2_biases",[depth], initializer=tf.contrib.layers.xavier_initializer())

      #128
      layer3_weights = tf.get_variable("layer3_weights", [patch_size_3, patch_size_3, depth, depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer3_biases = tf.get_variable("layer3_biases",[depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer4_weights = tf.get_variable("layer4_weights", [patch_size_3, patch_size_3, depth * 2, depth * 2], initializer=tf.contrib.layers.xavier_initializer())
      layer4_biases = tf.get_variable("layer4_biases",[depth * 2], initializer=tf.contrib.layers.xavier_initializer())

      #256
      layer5_weights = tf.get_variable("layer5_weights", [patch_size_3, patch_size_3, depth * 2, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer5_biases = tf.get_variable("layer5_biases",[depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer6_weights = tf.get_variable("layer6_weights", [patch_size_3, patch_size_3, depth * 4, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer6_biases = tf.get_variable("layer6_biases", [depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer7_weights = tf.get_variable("layer7_weights", [patch_size_3, patch_size_3, depth * 4, depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      layer7_biases = tf.get_variable("layer7_biases",[depth * 4], initializer=tf.contrib.layers.xavier_initializer())
      
      #512
      layer8_weights = tf.get_variable("layer8_weights", [patch_size_3, patch_size_3, depth * 4, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer8_biases = tf.get_variable("layer8_biases",[depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer9_weights = tf.get_variable("layer9_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer9_biases = tf.get_variable("layer9_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer10_weights = tf.get_variable("layer10_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer10_biases = tf.get_variable("layer10_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      
      fc = 9216 
      layer11_weights = tf.get_variable("layer11_weights", [fc, 4096], initializer=tf.contrib.layers.xavier_initializer())
      layer11_biases = tf.get_variable("layer11_biases", [4096], initializer=tf.contrib.layers.xavier_initializer())
      
      layer12_weights = tf.get_variable("layer12_weights", [4096, num_labels], initializer=tf.contrib.layers.xavier_initializer())
      layer12_biases = tf.get_variable("layer12_biases", [num_labels], initializer=tf.contrib.layers.xavier_initializer())

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

        #Conv->Relu->Conv->Relu->Conv->Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer5_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer5_biases)
        conv = tf.nn.conv2d(hidden, layer6_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer6_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(hidden, layer7_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer7_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(hidden, layer8_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer8_biases)
        conv = tf.nn.conv2d(hidden, layer9_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer9_biases)
        conv = tf.nn.conv2d(pool_1, layer10_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer10_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop = tf.nn.dropout(pool_1, keep_prob)
        shape = drop.get_shape().as_list()
        reshape = tf.reshape(drop, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.matmul(reshape, layer11_weights) + layer11_biases 
        drop = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(drop, layer12_weights) + layer12_biases 

      def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
      
      # Training computation.
      logits = model(tf_train_dataset, 0.5)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        
      tf.summary.scalar("Loss", loss)

      # Optimizer.
      optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))

      num_steps = 60001

    with tf.Session(graph=graph) as session:
      merged = tf.summary.merge_all()
      writer = tf.summary.FileWriter('./train', session.graph)

      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

        if (step % 250 == 0):
          _, l, predictions, m = session.run([optimizer, loss, train_prediction, merged], feed_dict=feed_dict)
          writer.add_summary(m, step)
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Valid accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
        else:
          #Don't pass in merged dictionary for better performance
          _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)


      print('Valid accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
      
      #Save session
      saver = tf.train.Saver()
      save_path = saver.save(session, model_save_path)
