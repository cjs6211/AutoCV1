# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops

np.random.seed(42)
tf.logging.set_verbosity(tf.logging.ERROR)
initializer = tf.contrib.layers.xavier_initializer()
#initializer = tf.initializers.he_normal()

class Model(object):
  """Construct a model with 3D CNN for classification."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_training = False
    self.metadata = metadata

    # Get the output dimension, i.e. number of classes
    self.output_dim = self.metadata.get_output_size()
    # Set batch size (for both training and testing)
    self.batch_size = 16
    #self.batch_size = 128

    # Get model function from class method below
    model_fn = self.model_fn
    # Classifier using model_fn
    self.classifier = tf.estimator.Estimator(model_fn=model_fn)

    # Attributes for preprocessing
    self.default_image_size = (128,128)
    self.default_num_frames = 10
    self.default_shuffle_buffer = 100

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.birthday = time.time()
    self.total_train_time = 0
    self.cumulated_num_steps = 0
    self.estimated_time_per_step = None
    self.total_test_time = 0
    self.cumulated_num_tests = 0
    self.estimated_time_test = None
    # Critical number for early stopping
    self.num_epochs_we_want_to_train = 5

  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    ****************************************************************************
    ****************************************************************************
    IMPORTANT: the loop of calling `train` and `test` will only run if
        self.done_training = False
      (the corresponding code can be found in ingestion.py, search
      'M.done_training')
      Otherwise, the loop will go on until the time budget is used up. Please
      pay attention to set self.done_training = True when you think the model is
      converged or when there is not enough time for next round of training.
    ****************************************************************************
    ****************************************************************************

    Args:
      dataset: a `tf.data.Dataset` object. Each of its examples is of the form
            (example, labels)
          where `example` is a dense 4-D Tensor of shape
            (sequence_size, row_count, col_count, num_channels)
          and `labels` is a 1-D Tensor of shape
            (output_dim,).
          Here `output_dim` represents number of classes of this
          multilabel classification task.

          IMPORTANT: some of the dimensions of `example` might be `None`,
          which means the shape on this dimension might be variable. In this
          case, some preprocessing technique should be applied in order to
          feed the training of a neural network. For example, if an image
          dataset has `example` of shape
            (1, None, None, 3)
          then the images in this datasets may have different sizes. On could
          apply resizing, cropping or padding in order to have a fixed size
          input tensor.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """
    

    # Count examples on training set
    
    """
    if not hasattr(self, 'num_examples_train'):
      logger.info("Counting number of examples on train set.")
      #dataset1 = dataset.batch(batch_size = self.batch_size)
      #dataset1 = dataset1.prefetch(1)
      imageset = dataset.map(lambda *x: x[1])
      imageset = imageset.batch(batch_size = 10000)
      iterator = imageset.make_one_shot_iterator()
      #print("number of train data {}".format(dataset.size()))
      #example, labels = iterator.get_next()
      example = iterator.get_next()
      sample_count = 0
      config = tf.ConfigProto()
      #num_cpu_core = os.cpu_count()
      #logger.info("number of cpu cores: {}".format(num_cpu_core))
      config.intra_op_parallelism_threads = 0
      config.inter_op_parallelism_threads = 2
      config.allow_soft_placement = True
      sess = tf.Session(config = config)
      #with tf.Session(config = config) as sess:
      #with tf.Session() as sess:
      while True:
        try:
          #sess.run(labels)
          sample_count += len(sess.run(example))
        except tf.errors.OutOfRangeError:
          break
      self.num_examples_train = sample_count
      logger.info("Finished counting. There are {} examples for training set."\
                  .format(sample_count))
    """
    
    #"""
    if not hasattr(self, 'num_examples_train'):
      logger.info("Counting number of examples on train set.")
      sample_count = self.metadata.size()
      self.num_examples_train = sample_count
      logger.info("Finished counting. There are {} examples for training set."\
                  .format(sample_count))
    #"""
    
    # Get number of steps to train according to some strategy
    steps_to_train = self.get_steps_to_train(remaining_time_budget)
    
    if steps_to_train <= 0:
      logger.info("Not enough time remaining for training. " +
            "Estimated time for training per step: {:.2f}, "\
            .format(self.estimated_time_per_step) +
            "but remaining time budget is: {:.2f}. "\
            .format(remaining_time_budget) +
            "Skipping...")
      self.done_training = True
    elif self.choose_to_stop_early():
      logger.info("The model chooses to stop further training because " +
                  "The preset maximum number of epochs for training is " +
                  "obtained: self.num_epochs_we_want_to_train = " +
                  str(self.num_epochs_we_want_to_train))
      self.done_training = True
    else:
      msg_est = ""
      if self.estimated_time_per_step:
        msg_est = "estimated time for this: {:.2f} sec."\
                  .format(steps_to_train * self.estimated_time_per_step)
      logger.info("Begin training for another {} steps...{}"\
                  .format(steps_to_train, msg_est))

      # Prepare input function for training
      train_input_fn = lambda: self.input_function(dataset, is_training=True)

      # Start training
      train_start = time.time()
      self.classifier.train(input_fn=train_input_fn, steps=steps_to_train)
      train_end = time.time()

      # Update for time budget managing
      train_duration = train_end - train_start
      self.total_train_time += train_duration
      self.cumulated_num_steps += steps_to_train
      self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
      logger.info("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) +\
            "Now total steps trained: {}. ".format(self.cumulated_num_steps) +\
            "Total time used for training: {:.2f} sec. ".format(self.total_train_time) +\
            "Current estimated time per step: {:.2e} sec.".format(self.estimated_time_per_step))

  def test(self, dataset, remaining_time_budget=None):
    """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """
    test_begin = time.time()
    logger.info("Begin testing... ")
    
    """
    # Count examples on test set
    if not hasattr(self, 'num_examples_test'):
      logger.info("Counting number of examples on test set.")
      iterator = dataset.make_one_shot_iterator()
      example, labels = iterator.get_next()
      sample_count = 0
      with tf.Session() as sess:
        while True:
          try:
            sess.run(labels)
            sample_count += 1
          except tf.errors.OutOfRangeError:
            break
      self.num_examples_test = sample_count
      logger.info("Finished counting. There are {} examples for test set."\
                  .format(sample_count))
    """
    

    # Prepare input function for testing
    test_input_fn = lambda: self.input_function(dataset, is_training=False)

    # Start testing (i.e. making prediction on test set)
    test_results = self.classifier.predict(input_fn=test_input_fn)

    predictions = [x['probabilities'] for x in test_results]
    predictions = np.array(predictions)
    test_end = time.time()
    # Update some variables for time management
    test_duration = test_end - test_begin
    self.total_test_time += test_duration
    self.cumulated_num_tests += 1
    self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
    logger.info("Successfully made one prediction. {:.2f} sec used. ".format(test_duration) +\
          "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) +\
          "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
    return predictions

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  # Model functions that contain info on neural network architectures
  # Several model functions are to be implemented, for different domains
  def conv3d_with_batchnorm(self, features, depth, kernel_size, padding, mode, layer_order):
    feature_map = tf.layers.conv3d(inputs=features, 
                                   filters=depth,
                                   kernel_size = kernel_size,
                                   kernel_initializer = initializer,
                                   padding = padding,
                                   use_bias = False)
    
    feature_map = tf.layers.batch_normalization(inputs=feature_map, 
                                                momentum=0.99, 
                                                training=mode)
    return prelu(feature_map, layer_order)

  def conv3d_without_batchnorm(self, features, depth, kernel_size, padding, mode, layer_order):
    feature_map = tf.layers.conv3d(inputs=features, 
                                   filters=depth,
                                   kernel_size = kernel_size,
                                   kernel_initializer = initializer,
                                   padding = padding,
                                   use_bias = True)
    
    return prelu(feature_map, layer_order)
    
  def model_fn(self, features, labels, mode):
    """Auto-Scaling 3D CNN model.

    For more information on how to write a model function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
    """
    input_layer = features*2.0-1.0

    # Replace missing values by 0
    hidden_layer = tf.where(tf.is_nan(input_layer),
                           tf.zeros_like(input_layer), input_layer)
    ###
    kernel_size = [1, 3, 3]
    hidden_layer_0_0 = self.conv3d_with_batchnorm(features = hidden_layer, 
                                                  depth = 16, 
                                                  kernel_size = kernel_size, 
                                                  padding = 'same', 
                                                  mode = mode == tf.estimator.ModeKeys.TRAIN, 
                                                  layer_order = 0)
    
    
    hidden_layer_0_1 = self.conv3d_with_batchnorm(features = hidden_layer, 
                                                     depth = 16, 
                                                     kernel_size = [1, 1, 1], 
                                                     padding = 'same', 
                                                     mode = mode == tf.estimator.ModeKeys.TRAIN, 
                                                     layer_order = 1)
    
    hideen_layer = tf.concat([hidden_layer_0_0,hidden_layer_0_1], axis = 4)
    
    ###
    # Repeatedly apply 3D CNN, followed by 3D max pooling
    # until the hidden layer has reasonable number of entries
    REASONABLE_NUM_ENTRIES = 1000
    num_filters = 16 # The number of filters is fixed
    i = 2
    while True:
      shape = hidden_layer.shape
      kernel_size = [min(3, shape[1]), min(3, shape[2]), min(3, shape[3])]
      hidden_layer_2 = tf.layers.conv3d(inputs=hidden_layer,
                                      filters=num_filters,
                                      kernel_size=kernel_size,
                                      kernel_initializer = initializer,
                                      use_bias = False,
                                      padding = 'same'
                                     )
      hidden_layer_2 = tf.layers.batch_normalization(inputs=hidden_layer_2, momentum=0.99, 
                                                   training=mode == tf.estimator.ModeKeys.TRAIN)
      hidden_layer_2 = prelu(hidden_layer_2, i)
      i += 1
      hidden_layer = tf.concat([hidden_layer, hidden_layer_2], axis=4)
      
      pool_size = [min(2, shape[1]), min(2, shape[2]), min(2, shape[3])]
      hidden_layer= tf.layers.max_pooling3d(inputs=hidden_layer,
                                            pool_size=pool_size,
                                            strides=pool_size,
                                            padding='valid',
                                            data_format='channels_last')
      
      if get_num_entries(hidden_layer) < REASONABLE_NUM_ENTRIES:
        break

    hidden_layer = tf.layers.flatten(hidden_layer)
    hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                   units=1024, 
                                   kernel_initializer = initializer,
                                   use_bias = False,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
                                   #activation=tf.nn.relu
                                  )
    hidden_layer = tf.layers.batch_normalization(inputs=hidden_layer, momentum=0.99,
                                                 training=mode == tf.estimator.ModeKeys.TRAIN)
    hidden_layer = tf.nn.relu(hidden_layer)
    hidden_layer = tf.layers.dropout(
        inputs=hidden_layer, rate=0.2,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=hidden_layer, units=self.output_dim,
                             kernel_initializer = initializer,
                             bias_initializer = initializer,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(0.001)
                             )
    sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # "classes": binary_predictions,
      # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": sigmoid_tensor
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # For multi-label classification, a correct loss is sigmoid cross entropy
    loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
      train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
      train_op = tf.group([train_op, update_ops])
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  def input_function(self, dataset, is_training):
    """Given `dataset` received by the method `self.train` or `self.test`,
    prepare input to feed to model function.

    For more information on how to write an input function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function
    """
    dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))

    if is_training:
      # Shuffle input examples
      dataset = dataset.shuffle(buffer_size=self.default_shuffle_buffer)
      # Convert to RepeatDataset to train for several epochs
      dataset = dataset.repeat()

    # Set batch size
    dataset = dataset.batch(batch_size=self.batch_size)
    dataset = dataset.prefetch(buffer_size = self.batch_size)
    
    """
    augmentation = [flip]
    for f in augmentation:
      dataset = dataset.map(lambda *x: (tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x[0]), lambda: x[0]), x[1]), 
                            num_parallel_calls=4)
    dataset = dataset.map(lambda *x: (tf.clip_by_value(x[0], 0, 1), x[1]))
    """    

    iterator = dataset.make_one_shot_iterator()
    example, labels = iterator.get_next()
    return example, labels

  def preprocess_tensor_4d(self, tensor_4d):
    """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
    non-fixed). The output tensor wil have fixed, known shape.

    Args:
      tensor_4d: A Tensor of shape
          [sequence_size, row_count, col_count, num_channels]
          where some dimensions might be `None`.
    Returns:
      A 4-D Tensor with fixed, known shape.
    """
    tensor_4d_shape = tensor_4d.shape
    logger.info("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

    if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
      num_frames = tensor_4d_shape[0]
    else:
      num_frames = self.default_num_frames
    if tensor_4d_shape[1] > 0:
      new_row_count = tensor_4d_shape[1]
    else:
      new_row_count=self.default_image_size[0]
    if tensor_4d_shape[2] > 0:
      new_col_count = tensor_4d_shape[2]
    else:
      new_col_count=self.default_image_size[1]

    """
    if not tensor_4d_shape[0] > 0:
      logger.info("Detected that examples have variable sequence_size, will " +
                "randomly crop a sequence with num_frames = " +
                "{}".format(num_frames))
      tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
    """
    if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0:
      logger.info("Detected that examples have variable space size, will " +
                "resize space axes to (new_row_count, new_col_count) = " +
                "{}".format((new_row_count, new_col_count)))
      if not self.default_image_size[0]==tensor_4d_shape[1] or not self.default_image_size[1]==tensor_4d_shape[2]: 
        tensor_4d = resize_space_axes(tensor_4d,
                                    new_row_count=new_row_count,
                                    new_col_count=new_col_count)
    logger.info("Tensor shape after preprocessing: {}".format(tensor_4d.shape))
    return tensor_4d

  def get_steps_to_train(self, remaining_time_budget):
    """Get number of steps for training according to `remaining_time_budget`.

    The strategy is:
      1. If no training is done before, train for 10 steps (ten batches);
      2. Otherwise, estimate training time per step and time needed for test,
         then compare to remaining time budget to compute a potential maximum
         number of steps (max_steps) that can be trained within time budget;
      3. Choose a number (steps_to_train) between 0 and max_steps and train for
         this many steps. Double it each time.
    """    
    
    
    if not remaining_time_budget: # This is never true in the competition anyway
      remaining_time_budget = 1200 # if no time limit is given, set to 20min

    if not self.estimated_time_per_step:
      steps_to_train = 200
      calculated_epochs = steps_to_train * self.batch_size / self.num_examples_train
      if calculated_epochs > self.num_epochs_we_want_to_train:
        self.num_epochs_we_want_to_train *= 2 
      #steps_to_train = int(ratio * self.num_examples_train/self.batch_size)
    else:
      if self.estimated_time_test:
        tentative_estimated_time_test = self.estimated_time_test
      else:
        tentative_estimated_time_test = 50 # conservative estimation for test
      max_steps = int((remaining_time_budget - tentative_estimated_time_test) / self.estimated_time_per_step)
      max_steps = max(max_steps, 1)
      if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
        #steps_to_train = int(2 ** self.cumulated_num_tests) # Double steps_to_train after each test
        #steps_to_train = int(2**self.cumulated_num_tests*20)
        #ratio = self.num_examples_test/self.num_examples_train
        #steps_to_train = int(ratio * self.num_examples_train/self.batch_size) 
        steps_to_train = int(self.num_examples_train/self.batch_size * 0.2 * self.cumulated_num_tests)
      else:
        steps_to_train = 0
        
    #num_epochs = self.cumulated_num_steps * self.batch_size / self.num_examples_train
    #if num_epochs > 1:
    #    steps_to_train = int(self.num_examples_train/self.batch_size)
    
    #steps_to_train = int(self.num_examples_train/self.batch_size)
    #steps_to_train = int(0.5 * self.num_examples_train/self.batch_size)    
    #steps_to_train = 100
    return steps_to_train 

  def age(self):
    return time.time() - self.birthday

  def choose_to_stop_early(self):
    """The criterion to stop further training (thus finish train/predict
    process).
    """
    batch_size = self.batch_size
    num_examples = self.num_examples_train
    num_epochs = self.cumulated_num_steps * batch_size / num_examples
    logger.info("Model already trained for {} epochs.".format(num_epochs))
    return num_epochs > self.num_epochs_we_want_to_train # Train for at least certain number of epochs then stop

def prelu(_x, name):
  var_name = "alpha_{}".format(name) 
  alphas = tf.get_variable(var_name, _x.get_shape()[-1],
                       initializer=tf.constant_initializer(1.0), trainable=True,
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
  """Re-implementation of this function:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

  Let z = labels, x = logits, then return the sigmoid cross entropy
    max(x, 0) - x * z + log(1 + exp(-abs(x)))
  (Then sum over all classes.)
  """
  labels = tf.cast(labels, dtype=tf.float32)
  relu_logits = tf.nn.relu(logits)
  exp_logits = tf.exp(- tf.abs(logits))
  sigmoid_logits = tf.log(1 + exp_logits)
  element_wise_xent = relu_logits - labels * logits + sigmoid_logits
  return tf.reduce_sum(element_wise_xent)

def get_num_entries(tensor):
  """Return number of entries for a TensorFlow tensor.

  Args:
    tensor: a tf.Tensor or tf.SparseTensor object of shape
        (batch_size, sequence_size, row_count, col_count[, num_channels])
  Returns:
    num_entries: number of entries of each example, which is equal to
        sequence_size * row_count * col_count [* num_channels]
  """
  tensor_shape = tensor.shape
  assert(len(tensor_shape) > 1)
  num_entries  = 1
  for i in tensor_shape[1:]:
    num_entries *= int(i)
  return num_entries

def crop_time_axis(tensor_4d, num_frames, begin_index=None):
  """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
  """
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[1], 0)
  padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

  # If not given, randomly choose the beginning index of frames
  if not begin_index:
    maxval = tf.shape(padded_tensor)[0] - num_frames + 1
    begin_index = tf.random.uniform([1],
                                    minval=0,
                                    maxval=maxval,
                                    dtype=tf.int32)
    begin_index = tf.stack([begin_index[0], 0, 0, 0], name='begin_index')

  sliced_tensor = tf.slice(padded_tensor,
                           begin=begin_index,
                           size=[num_frames, -1, -1, -1])

  return sliced_tensor

def resize_space_axes(tensor_4d, new_row_count, new_col_count):
  """Given a 4-D tensor, resize space axes to have target size.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels].
    new_row_count: An integer indicating the target row count.
    new_col_count: An integer indicating the target column count.
  Returns:
    A Tensor of shape [sequence_size, target_row_count, target_col_count].
  """
  resized_images = tf.image.resize_images(tensor_4d,
                                          size=(new_row_count, new_col_count))
  return resized_images

def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.WARNING)
  stderr_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger('INFO')


def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    batch_size = array_ops.shape(x)[0]
    uniform_random_left_right = random_ops.random_uniform([batch_size], 0, 1.0)
    flips_left_right = math_ops.round(array_ops.reshape(uniform_random_left_right, [batch_size, 1, 1, 1, 1]))
    flips_left_right = math_ops.cast(flips_left_right, x.dtype)
    flipped_input_left_right = array_ops.reverse(x, [3])
    output_image = flips_left_right * flipped_input_left_right + (1 - flips_left_right) * x
    
    uniform_random_up_down = random_ops.random_uniform([batch_size], 0, 1.0)
    flips_up_down = math_ops.round(array_ops.reshape(uniform_random_up_down, [batch_size, 1, 1, 1, 1]))
    flips_up_down = math_ops.cast(flips_up_down, x.dtype)
    flipped_input_up_down = array_ops.reverse(output_image, [2])
    
    x = flips_up_down * flipped_input_up_down + (1 - flips_up_down) * output_image
    
    #x = tf.image.random_flip_left_right(x)
    #x = tf.image.random_flip_up_down(x)

    return x

"""
def color(x: tf.Tensor) -> tf.Tensor:
    #Color augmentation

    #Args:
    #    x: Image

    #Returns:
    #    Augmented image
    
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x
"""

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

"""
def zoom(x: tf.Tensor) -> tf.Tensor:
    #Zoom augmentation

    #Args:
    #    x: Image

    #Returns:
    #    Augmented image
    

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes,box_ind=np.zeros(len(scales)),crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
"""