#%%
# 2.1 Import libraries.
import math
import gc
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

#%%
# 2.2 Define some constants.
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Batch size. Must be evenly dividable by dataset sizes.
BATCH_SIZE = 100
EVAL_BATCH_SIZE = 1

# Number of units in hidden layers.
HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 32

# Maximum number of training steps.
MAX_STEPS = 4000

# Directory to put the training data.
TRAIN_DIR = "data/train.csv"
MODEL_SAVE_PATH = 'model'

#%%
# 2.3 Get input data: get the sets of images and labels for training, validation, and
# test on MNIST.
import pandas as pd
train_df=pd.read_csv(TRAIN_DIR, sep=',')

#%%
train_array = train_df.values
test_x_array = train_array[0:1000, 1:785]
test_y_array = train_array[0:1000, 0]
train_x_array = train_array[1000:42000, 1:785]
train_y_array = train_array[1000:42000, 0]

#%%
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

train_ds = DataSet(train_x_array, train_y_array)
test_ds = DataSet(test_x_array, test_y_array)

del train_array
del train_x_array
del train_y_array
del test_x_array
del test_y_array
gc.collect()


#%%
# 2.4 Build inference graph.
def mnist_inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
    Returns:
        logits: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    # Uncomment the following line to see what we have constructed.
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),
                         "model", "inference.pbtxt", as_text=True)
    return logits

#%%
# 2.5 Build training graph.
def mnist_training(logits, labels, learning_rate):
    """Build the training graph.

    Args:
        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
        labels: Labels tensor, int32 - [BATCH_SIZE], with values in the
          range [0, NUM_CLASSES).
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    tf.summary.scalar('loss', loss)
    
    return train_op, loss

#%%
# 2.6 Build the complete graph for feeding inputs, training, and saving checkpoints.
mnist_graph = tf.Graph()
with mnist_graph.as_default():
    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32)                                       
    labels_placeholder = tf.placeholder(tf.int32)
    tf.add_to_collection("images", images_placeholder)  # Remember this Op.
    tf.add_to_collection("labels", labels_placeholder)  # Remember this Op.

    # Build a Graph that computes predictions from the inference model.
    logits = mnist_inference(images_placeholder,
                             HIDDEN1_UNITS,
                             HIDDEN2_UNITS)
    tf.add_to_collection("logits", logits)  # Remember this Op.

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op, loss = mnist_training(logits, labels_placeholder, 0.01)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    


#%%
# 2.7 Run training for MAX_STEPS and save checkpoint at the end.
with tf.Session(graph=mnist_graph) as sess:
    
    train_writer = tf.summary.FileWriter('model',sess.graph)
    
    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(MAX_STEPS):
        # Read a batch of images and labels.
        images_feed, labels_feed = train_ds.next_batch(BATCH_SIZE)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        run_metadata = tf.RunMetadata()
        summary, loss_value = sess.run([train_op, loss],
                                 feed_dict={images_placeholder: images_feed,
                                            labels_placeholder: labels_feed},
                                 run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        train_writer.add_summary(summary, step)

        # Print out loss value.
        if step % 500 == 0:
            print('Step %d: loss = %.2f' % (step, loss_value))

    train_writer.close()
    # Write a checkpoint.
    checkpoint_file = os.path.join(MODEL_SAVE_PATH, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=step)
    
#%%
# 2.8 Run evaluation based on the saved checkpoint. (One example)
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(MODEL_SAVE_PATH, "checkpoint-1999.meta"))
    saver.restore(
        sess, os.path.join(MODEL_SAVE_PATH, "checkpoint-1999"))

    # Retrieve the Ops we 'remembered'.
    logits = tf.get_collection("logits")[0]
    images_placeholder = tf.get_collection("images")[0]
    labels_placeholder = tf.get_collection("labels")[0]
    
    # Add an Op that chooses the top k predictions.
    eval_op = tf.nn.top_k(logits)
    
    # Run evaluation.
    images_feed, labels_feed = test_ds.next_batch(EVAL_BATCH_SIZE)
    imgplot = plt.imshow(np.reshape(images_feed, (28, 28)))
    prediction = sess.run(eval_op,
                          feed_dict={images_placeholder: images_feed,
                                     labels_placeholder: labels_feed})
    print("Ground truth: %d\nPrediction: %d" % (labels_feed, prediction.indices[0][0]))

#%%
# 2.8 Run evaluation based on the saved checkpoint. (All test_set)
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(MODEL_SAVE_PATH, "checkpoint-3999.meta"))
    saver.restore(
        sess, os.path.join(MODEL_SAVE_PATH, "checkpoint-3999"))

    # Retrieve the Ops we 'remembered'.
    logits = tf.get_collection("logits")[0]
    images_placeholder = tf.get_collection("images")[0]
    labels_placeholder = tf.get_collection("labels")[0]
    
    # Add an Op that chooses the top k predictions.
    eval_op = tf.nn.top_k(logits)
    
    # Run evaluation.
    accuracy = np.array([0] * test_ds.num_examples)
    for i in xrange(test_ds.num_examples):        
        images_feed, labels_feed = test_ds.next_batch(EVAL_BATCH_SIZE)
        prediction = sess.run(eval_op,
                              feed_dict={images_placeholder: images_feed,
                                         labels_placeholder: labels_feed})
        if labels_feed == prediction.indices[0][0]:
            accuracy[i] = 1

    # Show results.
    a = np.sum(accuracy)
    print (float(a)/test_ds.num_examples)


    
