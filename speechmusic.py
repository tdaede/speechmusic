#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["features_trans5/data"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

print(tf.decode_csv(value,record_defaults=[[0.0] for x in range(0,28)],field_delim=' '))