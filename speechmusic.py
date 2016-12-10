#!/usr/bin/env python

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()


filename_queue = tf.train.string_input_producer(["features_trans5/data"])
threads = tf.train.start_queue_runners()

# decode_csv treats extra delimiters as extra fields, the training data has extra spaces
# between input and output. so column 27 and 28 are the actual outputs

reader = tf.TextLineReader()
key, value = reader.read_up_to(filename_queue,1000) # read a batch of 100
csv = tf.decode_csv(value,record_defaults=[[0.0] for x in range(0,29)],field_delim=' ')

training = tf.pack(csv[0:25],axis=1)
target = tf.pack(csv[27:28],axis=1)


#x = tf.placeholder(tf.float32, shape=[None, 25])
#y_ = tf.placeholder(tf.float32, shape=[None, 2])

x = training
y_ = target

#print(target.eval())

W = tf.Variable(tf.zeros([25,1]))
b = tf.Variable(tf.zeros([1]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

mse = tf.reduce_mean(tf.abs(tf.subtract(y, y_)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(mse)

for i in range(1000):
    # run on the entire set, rip your cpu
    train_step.run()
    #print(y_.eval())

correct_prediction = tf.abs(tf.subtract(tf.minimum(tf.maximum(y,-1),1),y_))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(W.eval())
print(b.eval())
print(accuracy.eval())

