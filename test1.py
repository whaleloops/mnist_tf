import numpy as np
import tensorflow as tf
import gc

class DataSet(object):
    def __init__(self,images,labels,dtype=np.int64):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.int64, np.float32):
          raise TypeError('Invalid image dtype %r, expected np.int64 or np.float32' %
                          dtype)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]


        if dtype == np.int64:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

import pandas as pd

TRAIN_DIR = "data/train.csv"
EVAL_BATCH_SIZE = 1

print("Loading data")
train_df=pd.read_csv(TRAIN_DIR, sep=',', nrows=20)

#%%
train_array = train_df.values
test_x_array = train_array[0:10, 1:785]
test_y_array = train_array[0:10, 0]

test_ds = DataSet(test_x_array, test_y_array)

del test_x_array
del test_y_array
gc.collect()

print("Start predicting")

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph("model/checkpoint-199.meta")
    saver.restore(sess, "model/checkpoint-199")

    # Retrieve the Ops we 'remembered'.
    logits = tf.get_collection("logits")[0]
    images_placeholder = tf.get_collection("images")[0]
    labels_placeholder = tf.get_collection("labels")[0]
    keep_prob = tf.get_collection("prop")[0]

    # Add an Op that chooses the top k predictions.
    eval_op = tf.nn.top_k(logits)

    # Run evaluation.
    accuracy = np.array([0] * test_ds.num_examples)
    for i in xrange(test_ds.num_examples):
        images_feed, labels_feed = test_ds.next_batch(EVAL_BATCH_SIZE)
        prediction = sess.run(eval_op,
                              feed_dict={images_placeholder: images_feed,
                                         labels_placeholder: labels_feed,
                                         keep_prob: 1})
        if labels_feed == prediction.indices[0][0]:
            accuracy[i] = 1
    a = np.sum(accuracy)
    print (float(a)/test_ds.num_examples)