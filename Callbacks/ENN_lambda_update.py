import tensorflow as tf
import os

class ENN_lambda_update(tf.keras.callbacks.Callback):
    def __init__(self, lambda_t=0, max_t=1):
        self.lambda_t = tf.Variable(initial_value=lambda_t, dtype=tf.float64)
        self.max_t = tf.Variable(initial_value=max_t, dtype=tf.float64)

    def on_epoch_end(self, epoch, logs={}):
        self.lambda_t.assign(tf.reduce_min([self.max_t, tf.cast(epoch, tf.dtypes.float64)/30.0]))

