import os
import sys
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

image_size = 96; # 150x150.  Also, 224, 96, 64, and 32 are also common
num_channels = 3
pixel_depth = 255.0  # Number of levels per pixel.
num_labels = 2
patch_size_3 = 3
depth = 32




if sys.platform == 'win32':
    os.chdir("C:\\git\\Kaggles\\CatVsDogs\\CatVsDogs")



# resizes to image_size/image_size while keeping aspect ratio the same.  pads on right/bottom as appropriate 
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    if (img.shape[0] >= img.shape[1]): # height is greater than width
       resizeto = (image_size, int (round (image_size * (float (img.shape[1])  / img.shape[0]))));
    else:
       resizeto = (int (round (image_size * (float (img.shape[0])  / img.shape[1]))), image_size);
    
    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.copyMakeBorder(img2, 0, image_size - img2.shape[0], 0, image_size - img2.shape[1], cv2.BORDER_CONSTANT, 0)

    return img3[:,:,::-1]  # turn into rgb format

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, image_size, image_size, num_channels), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = read_image(image_file);
        image_data = np.array (image, dtype=np.float32);
        image_data[:,:,0] = (image_data[:,:,0].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,1] = (image_data[:,:,1].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,2] = (image_data[:,:,2].astype(float) - pixel_depth / 2) / pixel_depth
        
        data[i] = image_data; # image_data.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))    
    return data




np.random.seed(42)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (labels=='cats').astype(np.float32); # set dogs to 0 and cats to 1
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels



#test_dataset, test_labels = reformat(test_dataset, test_labels)
#print ('Test set', test_dataset.shape, test_labels.shape)


def ConvNet(model_save_path):

    TRAINING_AND_VALIDATION_SIZE_DOGS = 12500
    TRAINING_AND_VALIDATION_SIZE_CATS = 12500 
    TRAINING_AND_VALIDATION_SIZE_ALL  = TRAINING_AND_VALIDATION_SIZE_CATS + TRAINING_AND_VALIDATION_SIZE_DOGS
    TRAINING_SIZE = TRAINING_AND_VALIDATION_SIZE_ALL  # TRAINING_SIZE + VALID_SIZE must equal TRAINING_AND_VALIDATION_SIZE_ALL
    VALID_SIZE = 400
    TEST_SIZE_ALL = 500

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] 
    train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

    train_images = train_dogs + train_cats
    train_labels = np.array ((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + (['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS))

    train_normalized = prep_data(train_images)
    print("Train shape: {}".format(train_normalized.shape))

    train_dataset_rand, train_labels_rand = randomize(train_normalized, train_labels)

    print ('Training set', train_dataset.shape, train_labels.shape)
    print ('Validation set', valid_dataset.shape, valid_labels.shape)

    # split up into training + valid
    valid_dataset = train_dataset_rand[:VALID_SIZE,:,:,:]
    valid_labels =   train_labels_rand[:VALID_SIZE]
    train_dataset = train_dataset_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE,:,:,:]
    train_labels  = train_labels_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE]
    print ('Training', train_dataset.shape, train_labels.shape)
    print ('Validation', valid_dataset.shape, valid_labels.shape)


    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

    batch_size = 16
    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
      
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
      
      layer7_weights = tf.get_variable("layer7_weights", [patch_size_3, patch_size_3, depth * 4, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer7_biases = tf.get_variable("layer7_biases",[depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer8_weights = tf.get_variable("layer8_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer8_biases = tf.get_variable("layer8_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      
      fc = 9216 
      layer9_weights = tf.get_variable("layer9_weights", [fc, 4096], initializer=tf.contrib.layers.xavier_initializer())
      layer9_biases = tf.get_variable("layer9_biases", [4096], initializer=tf.contrib.layers.xavier_initializer())
      
      layer10_weights = tf.get_variable("layer10_weights", [4096, num_labels], initializer=tf.contrib.layers.xavier_initializer())
      layer10_biases = tf.get_variable("layer10_biases", [num_labels], initializer=tf.contrib.layers.xavier_initializer())

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

        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer5_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer5_biases)
        conv = tf.nn.conv2d(hidden, layer6_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer6_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer7_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer7_biases)
        conv = tf.nn.conv2d(hidden, layer8_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer8_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop = tf.nn.dropout(pool_1, keep_prob)
        shape = drop.get_shape().as_list()
        reshape = tf.reshape(drop, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.matmul(reshape, layer9_weights) + layer9_biases 
        drop = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(drop, layer10_weights) + layer10_biases 

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


def LoadAndRun(model_save_path):
    
    test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
    test_data = prep_data(test_images)
    
    print("TEST IMAGES: ", len(test_images))
    print("TEST DATA: ", len(test_data))

    tf.reset_default_graph()
    graph = tf.Graph()
    num_steps = 125
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
      
      layer7_weights = tf.get_variable("layer7_weights", [patch_size_3, patch_size_3, depth * 4, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer7_biases = tf.get_variable("layer7_biases",[depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer8_weights = tf.get_variable("layer8_weights", [patch_size_3, patch_size_3, depth * 8, depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      layer8_biases = tf.get_variable("layer8_biases", [depth * 8], initializer=tf.contrib.layers.xavier_initializer())
      
      fc = 9216 
      layer9_weights = tf.get_variable("layer9_weights", [fc, 4096], initializer=tf.contrib.layers.xavier_initializer())
      layer9_biases = tf.get_variable("layer9_biases", [4096], initializer=tf.contrib.layers.xavier_initializer())
      
      layer10_weights = tf.get_variable("layer10_weights", [4096, num_labels], initializer=tf.contrib.layers.xavier_initializer())
      layer10_biases = tf.get_variable("layer10_biases", [num_labels], initializer=tf.contrib.layers.xavier_initializer())

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

        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer5_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer5_biases)
        conv = tf.nn.conv2d(hidden, layer6_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer6_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Conv->Relu->Conv-Relu->Pool
        conv = tf.nn.conv2d(pool_1, layer7_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer7_biases)
        conv = tf.nn.conv2d(hidden, layer8_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer8_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop = tf.nn.dropout(pool_1, keep_prob)
        shape = drop.get_shape().as_list()
        reshape = tf.reshape(drop, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.matmul(reshape, layer9_weights) + layer9_biases 
        drop = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(drop, layer10_weights) + layer10_biases 



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
