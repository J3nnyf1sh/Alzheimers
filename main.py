import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)


brain_tumour = tf.kera.datasets.brain_tumour

(train_images, train_labels), (valid_images, valid_labels) = brain_tumour.load_data()
train_images, test_images = train_images/512, valid_images/512