#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["features_trans5/data"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

csv = tf.decode_csv(value,record_defaults=[[0.0] for x in range(0,27)],field_delim=' ')
training = csv[0:25]
target = csv[25:27]

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 25])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.zeros([25,2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

#we don't want a softmax here but probably simple MSE, send help
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#for i in range(1000):
#  batch = mnist.train.next_batch(100)
#  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

