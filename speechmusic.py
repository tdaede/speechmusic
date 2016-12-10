#!/usr/bin/env python

import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["features_trans5/data"])

reader = tf.TextLineReader()
key, value = reader.read_up_to(filename_queue,100) # read a batch of 100
csv = tf.decode_csv(value,record_defaults=[[0.0] for x in range(0,29)],field_delim=' ')

training = tf.pack(csv[0:25],axis=1)
target = tf.pack(csv[25:27],axis=1)


sess = tf.InteractiveSession()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

#x = tf.placeholder(tf.float32, shape=[None, 25])
#y_ = tf.placeholder(tf.float32, shape=[None, 2])

x = training
y_ = target

W = tf.Variable(tf.zeros([25,2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

#we don't want a softmax here but probably simple MSE, send help
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    # run on the entire set, rip your cpu
    train_step.run()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval())

