# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from time import sleep


TRAINING_BATCH_SIZE = 20
TRAINING_BATCHES = 200

num = plt.subplot2grid((7, 5), (0,0), colspan=5, rowspan=5)
pws = [
    plt.subplot2grid((7, 5), (5,0)),
    plt.subplot2grid((7, 5), (5,1)),
    plt.subplot2grid((7, 5), (5,2)),
    plt.subplot2grid((7, 5), (5,3)),
    plt.subplot2grid((7, 5), (5,4)),
    plt.subplot2grid((7, 5), (6,0)),
    plt.subplot2grid((7, 5), (6,1)),
    plt.subplot2grid((7, 5), (6,2)),
    plt.subplot2grid((7, 5), (6,3)),
    plt.subplot2grid((7, 5), (6,4))
]

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(TRAINING_BATCHES):
    batch_xs, batch_ys = mnist.train.next_batch(TRAINING_BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    image = np.reshape(batch_xs[-1], (28, 28))
    ww = np.reshape(W.eval().swapaxes(1,0), (10, 28, 28))
    num.imshow(image)
    num.set_title("Latest Image Sample")
    num.axis("off")
    for i in range(10):
        pws[i].imshow(ww[i])
        pws[i].axis("off")
        pws[i].set_title(str(i))
    
    plt.draw()
    plt.pause(0.1)

    for i in range(10):
        pws[i].cla()
    num.cla()

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
