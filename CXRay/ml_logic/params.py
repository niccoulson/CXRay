
import os
import numpy as np


LR = 1e-5  # For transfer learning, use a small value such as 1e5
EPOCHS = 30

IMG_SIZE = 224  # img_dims[0]
CHANNELS = 3  # The images WERE grayscale but we converted them to 3-channel
opt = tf.keras.optimizers.Adam(learning_rate=LR)