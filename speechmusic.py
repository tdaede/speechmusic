#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import threading

config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, \
                        allow_soft_placement=True)
sess = tf.InteractiveSession(config=config)


filename_queue = tf.train.string_input_producer(["features_trans5/data"])
threads = tf.train.start_queue_runners()

# decode_csv treats extra delimiters as extra fields, the training data has extra spaces
# between input and output. so column 27 and 28 are the actual outputs

reader = tf.TextLineReader()
key, value = reader.read_up_to(filename_queue,10000) # read a batch of 100
csv = tf.decode_csv(value,record_defaults=[[0.0] for x in range(0,29)],field_delim=' ')

training = tf.pack(csv[0:25],axis=1)
target = tf.pack(csv[28:29],axis=1)


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
error = tf.abs(tf.subtract(tf.minimum(tf.maximum(y,-1),1),y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(error)

def train_function():
    for i in range(1000):
        # run on the entire set, rip your cpu
        train_step.run(session=sess)
        error = tf.abs(tf.subtract(tf.minimum(tf.maximum(y,-1),1),y_))
        percent_correct = tf.reduce_mean(tf.cast(tf.equal(tf.greater(y,0.0),tf.greater(y_,0.0)),tf.float32))
        accuracy = tf.reduce_mean(tf.cast(error, tf.float32))
        print(i,accuracy.eval(),percent_correct.eval())
        #print(y_.eval())

train_function()

# we do asynchronous parallel gradient descent because otherwise
# it doesn't parallelize very well
#train_threads = []
#for _ in range(4):
#    train_threads.append(threading.Thread(target=train_function))

# Start the threads, and block on their completion.
#for t in train_threads:
#    t.start()
#for t in train_threads:
#    t.join()

error = tf.abs(tf.subtract(tf.minimum(tf.maximum(y,-1),1),y_))
percent_correct = tf.reduce_mean(tf.cast(tf.equal(tf.greater(y,0.0),tf.greater(y_,0.0)),tf.float32))
accuracy = tf.reduce_mean(tf.cast(error, tf.float32))
print(W.eval())
print(b.eval())
print(percent_correct.eval())
#print(accuracy.eval())

