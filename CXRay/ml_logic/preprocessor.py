import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd

def expand_greyscale_image_channels(image):
    if image.shape[-1] == 1:
        grey_image_3_channel = tf.tile(image, tf.constant([1, 1, 3], tf.int32))
    else:
        grey_image_3_channel = image

    return preprocess_input(grey_image_3_channel)
