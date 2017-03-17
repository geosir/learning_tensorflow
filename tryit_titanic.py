# Build your very own Neural Network using this walkthrough!
# Based loosely the mnist_softmax.py from the TensorFlow tutorials.

import pandas as pa
import tensorflow as tf
import numpy as np
import random

training_csv = pa.read_csv("train.csv")
testing_csv = pa.read_csv("test.csv")

print(testing_csv)

for i, row in training_csv.iterrows():
    training_csv.set_value(i, "Sex", 1 if row["Sex"] == "male" else -1)
    training_csv.set_value(i, "Age", 100 if pa.isnull(row["Age"]) else row["Age"])

training_data = []
for i, row in training_csv.iterrows():
    entry = [row[column] for column in ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    training_data.append(entry)

for i, row in testing_csv.iterrows():
    training_csv.set_value(i, "Sex", 1 if row["Sex"] == "male" else -1)
    training_csv.set_value(i, "Age", 100 if pa.isnull(row["Age"]) else row["Age"])

testing_data = []
for i, row in training_csv.iterrows():
    entry = [row[column] for column in ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    testing_data.append(entry)

#print(training_data)

# Create the model
x = tf.placeholder(tf.float32, [None, 6])
W = tf.Variable(tf.zeros([6, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

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

def get_next_batch(count):
    batch_xs = []
    batch_ys = []
    for i in range(count):
        choice = random.choice(training_data)
        batch_ys.append([1, 0] if choice[0] == 1 else [0, 1])
        batch_xs.append(choice[1:])
    return batch_xs, batch_ys


# Train
for _ in range(len(training_data)):
    batch_xs, batch_ys = get_next_batch(10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
    #                                     y_: mnist.test.labels}))
