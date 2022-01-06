import tensorflow as tf

class Accuracy(tf.keras.metrics.Metric):

    def __init__(self, num_classes, split_input=False, name='accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', shape=(1,), initializer='zeros', dtype=tf.float32)
        self.counter = self.add_weight(name="counter", shape=(1,), initializer="zeros", dtype=tf.float32)

        self.num_classes = num_classes
        self.epsilon = 10e-10
        self.split_input = split_input

    def update_state(self, y_true, y_pred, sample_weight=None):
            
        if self.num_classes == 2:

            self.tp.assign_add([
                  tf.reduce_sum(y_true*tf.math.round(tf.keras.activations.sigmoid(y_pred)) + (1-y_true)*(1-tf.math.round(tf.keras.activations.sigmoid(y_pred))), axis=(0,1))])
 
            self.counter.assign_add([tf.shape(y_true)[0]])
        else:
            if self.split_input:
                y_true_b = tf.split(y_true, num_or_size_splits=2, axis=0)[0]
                y_pred_b = tf.split(y_pred, num_or_size_splits=2, axis=0)[0]
            else:
                y_true_b = y_true
                y_pred_b = y_pred

            y_pred_one_hot = tf.one_hot(tf.argmax(y_pred_b, axis=-1), depth=self.num_classes)
            self.counter.assign_add([tf.reduce_sum(y_true_b, axis=(0,1))])
            self.tp.assign_add([tf.reduce_sum(y_true_b * y_pred_one_hot, axis=(0,1))])

    def result(self):
        return self.tp / self.counter

    def reset_states(self):
        self.tp.assign([0])
        self.counter.assign([0])
    

class AverageAccuracy(tf.keras.metrics.Metric):

    def __init__(self, num_classes, split_input=False, name='avg_accuracy', **kwargs):
        super(AverageAccuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.class_count = self.add_weight(name="counter", shape=(num_classes,), initializer="zeros", dtype=tf.float32)

        self.num_classes = num_classes
        self.split_input = split_input
        self.epsilon = 10e-10

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.split_input:
            y_true_b = tf.split(y_true, num_or_size_splits=2, axis=0)[0]
            y_pred_b = tf.split(y_pred, num_or_size_splits=2, axis=0)[0]
        else:
            y_true_b = y_true
            y_pred_b = y_pred

        y_pred_one_hot = tf.one_hot(tf.argmax(y_pred_b, axis=-1), depth=self.num_classes)
        self.class_count.assign_add(tf.reduce_sum(y_true_b, axis=0))
        self.tp.assign_add(tf.reduce_sum(y_true_b * y_pred_one_hot, axis=0))

    def result(self):
        return tf.reduce_mean(self.tp / (self.class_count + 0.0000001))

    def reset_states(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.class_count.assign(tf.zeros_like(self.class_count))


class F1_score(tf.keras.metrics.Metric):

    def __init__(self, num_classes, split_input=False, name='f1_score', **kwargs):
        super(F1_score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.tn = self.add_weight(name='tn', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32)

        self.num_classes = num_classes
        self.split_input = split_input
        self.epsilon = 10e-10

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.split_input:
            y_true_b = tf.split(y_true, num_or_size_splits=2, axis=0)[0]
            y_pred_b = tf.split(y_pred, num_or_size_splits=2, axis=0)[0]
        else:
            y_true_b = y_true
            y_pred_b = y_pred

        y_true_b = tf.cast(y_true_b, tf.bool)
        y_pred_b = tf.cast(tf.one_hot(tf.argmax(y_pred_b, axis=-1), depth=self.num_classes), tf.bool)

        tp_batch = tf.cast(tf.logical_and(tf.equal(y_true_b, True), tf.equal(y_pred_b, True)), tf.float32)
        fp_batch = tf.cast(tf.logical_and(tf.equal(y_true_b, False), tf.equal(y_pred_b, True)), tf.float32)
        tn_batch = tf.cast(tf.logical_and(tf.equal(y_true_b, False), tf.equal(y_pred_b, False)), tf.float32)
        fn_batch = tf.cast(tf.logical_and(tf.equal(y_true_b, True), tf.equal(y_pred_b, False)), tf.float32)

        self.tp.assign_add(tf.cond(tf.equal(tf.size(tp_batch),0), lambda: tf.zeros_like(self.tp), lambda: tf.reduce_sum(tp_batch, axis=0)))
        self.fp.assign_add(tf.cond(tf.equal(tf.size(fp_batch),0), lambda: tf.zeros_like(self.fp), lambda: tf.reduce_sum(fp_batch, axis=0)))
        self.tn.assign_add(tf.cond(tf.equal(tf.size(tn_batch),0), lambda: tf.zeros_like(self.tn), lambda: tf.reduce_sum(tn_batch, axis=0)))
        self.fn.assign_add(tf.cond(tf.equal(tf.size(fn_batch),0), lambda: tf.zeros_like(self.fn), lambda: tf.reduce_sum(fn_batch, axis=0)))

    def result(self):
        return tf.reduce_mean(self.result_class_wise())

    def result_class_wise(self):
        p = self.tp / (self.tp + self.fp + self.epsilon)
        r = self.tp / (self.tp + self.fn + self.epsilon)
        return 2*p*r / (p + r + self.epsilon)

    def reset_states(self):
        self.tp.assign(tf.zeros_like(self.tp, dtype=tf.float32))
        self.fp.assign(tf.zeros_like(self.fp, dtype=tf.float32))
        self.tn.assign(tf.zeros_like(self.tn, dtype=tf.float32))
        self.fn.assign(tf.zeros_like(self.fn, dtype=tf.float32))
