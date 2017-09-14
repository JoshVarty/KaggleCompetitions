import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from fgsm import fgsm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

img_size = 28
img_depth = 1
num_labels = 10

data = pd.read_csv("../input/train.csv");

#Partition into train/valid sets
images = data.iloc[:,1:]
labels = data.iloc[:,:1]
train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, train_size=0.9, random_state=0)
valid_images, test_images, valid_labels, test_labels = train_test_split(valid_images, valid_labels, train_size=0.5, random_state=0)

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, img_size, img_size, img_depth)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  labels = np.reshape(labels, (len(labels), num_labels))
  return dataset, labels

train_images, train_labels = reformat(train_images.as_matrix(), train_labels.as_matrix())
valid_images, valid_labels = reformat(valid_images.as_matrix(), valid_labels.as_matrix())
test_images, test_labels = reformat(test_images.as_matrix(), test_labels.as_matrix())

def model(x, logits=False, training=False):
    conv0 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same', name='conv0', activation=tf.nn.relu)
    pool0 = tf.layers.max_pooling2d(conv0, pool_size=[2, 2], strides=2, name='pool0')
    conv1 = tf.layers.conv2d(pool0, filters=64, kernel_size=[3, 3], padding='same', name='conv1', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
    flat = tf.reshape(pool1, [-1, 7*7*64], name='flatten')
    dense = tf.layers.dense(flat, units=128, activation=tf.nn.relu, name='dense')
    dropout = tf.layers.dropout(dense, rate=0.25, training=training, name='dropout')
    logits_ = tf.layers.dense(dropout, units=10, name='logits')

    y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y


# Collect all tensorflow tensors into one "enviroment" to avoid
# accidental overwriting.
class Dummy:
    pass
env = Dummy()

# We need a scope since the inference graph will be reused later
with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_depth), name='x')
    env.y = tf.placeholder(tf.float32, (None, num_labels), name='y')
    env.training = tf.placeholder(bool, (), name='mode')

    env.ybar, logits = model(env.x, logits=True,
                             training=env.training)

    z = tf.argmax(env.y, axis=1)
    zbar = tf.argmax(env.ybar, axis=1)
    count = tf.cast(tf.equal(z, zbar), tf.float32)
    env.acc = tf.reduce_mean(count, name='acc')

    xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y, logits=logits)
    env.loss = tf.reduce_mean(xent, name='loss')

env.optim = tf.train.AdamOptimizer().minimize(env.loss)

# Note the reuse=True flag
with tf.variable_scope('model', reuse=True):
    env.x_adv = fgsm(model, env.x, epochs=12, eps=0.02)

# --------------------------------------------------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# --------------------------------------------------------------------

def _evaluate(X_data, y_data, env):
    print('\nEvaluating')
    n_sample = X_data.shape[0]
    batch_size = 128
    n_batch = int(np.ceil(n_sample/batch_size))
    loss, acc = 0, 0
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
        start = ind*batch_size
        end = min(n_sample, start+batch_size)
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end],
                       env.training: False})
        loss += batch_loss*batch_size
        acc += batch_acc*batch_size
    loss /= n_sample
    acc /= n_sample
    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def _predict(X_data, env):
    print('\nPredicting')
    n_sample = X_data.shape[0]
    batch_size = 128
    n_batch = int(np.ceil(n_sample/batch_size))
    yval = np.empty((X_data.shape[0], num_labels))
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
        start = ind*batch_size
        end = min(n_sample, start+batch_size)
        batch_y = sess.run(env.ybar, feed_dict={
            env.x: X_data[start:end], env.training: False})
        yval[start:end] = batch_y
    print()
    return yval

# --------------------------------------------------------------------

print('\nTraining')
n_sample = train_images.shape[0]
batch_size = 128
n_batch = int(np.ceil(n_sample/batch_size))
n_epoch = 5
for epoch in range(n_epoch):
    print('Epoch {0}/{1}'.format(epoch+1, n_epoch))
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
        start = ind*batch_size
        end = min(n_sample, start+batch_size)
        sess.run(env.optim, feed_dict={env.x: train_images[start:end],
                                       env.y: train_labels[start:end],
                                       env.training: True})
    _evaluate(valid_images, valid_labels, env)

print('\nTesting against clean data')
_evaluate(test_images, test_labels, env)

# --------------------------------------------------------------------

if False:
    print('\nLoading adversarial')
    X_adv = np.load('data/ex_00.npy')
else:
    print('\nCrafting adversarial')
    n_sample = test_images.shape[0]
    batch_size = 128
    n_batch = int(np.ceil(n_sample/batch_size))
    n_epoch = 20
    X_adv = np.empty_like(test_images)
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind+1, n_batch), end='\r')
        start = ind*batch_size
        end = min(n_sample, start+batch_size)
        tmp = sess.run(env.x_adv, feed_dict={env.x: test_images[start:end],
                                             env.y: test_labels[start:end],
                                             env.training: False})
        X_adv[start:end] = tmp
    print('\nSaving adversarial')
    os.makedirs('data', exist_ok=True)
    np.save('data/ex_00.npy', X_adv)


print('\nTesting against adversarial data')
_evaluate(X_adv, test_labels, env)

# --------------------------------------------------------------------

y1 = _predict(test_images, env)
y2 = _predict(X_adv, env)

z0 = np.argmax(test_labels, axis=1)
z1 = np.argmax(y1, axis=1)
z2 = np.argmax(y2, axis=1)

X_tmp = np.empty((10, 28, 28))
y_tmp = np.empty((10, 10))
for i in range(10):
    print('Target {0}'.format(i))
    ind, = np.where(np.all([z0==i, z1==i, z2!=i], axis=0))
    cur = np.random.choice(ind)
    X_tmp[i] = np.squeeze(X_adv[cur])
    y_tmp[i] = y2[cur]

print('\nPlotting results')
fig = plt.figure(figsize=(10, 1.8))
gs = gridspec.GridSpec(1, 10, wspace=0.1, hspace=0.1)

label = np.argmax(y_tmp, axis=1)
proba = np.max(y_tmp, axis=1)
for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
                  fontsize=12)

print('\nSaving figure')
gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/ex_00.png')