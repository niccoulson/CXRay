
from CXRay.ml_logic.params import (LR,
                                   EPOCHS,
                                   IMG_SIZE,
                                   CHANNELS,
                                   opt
                                   )

from tensorflow.keras import model, Sequential, layers, regularizers, optimizers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from typing import Tuple

# Initialize pretrained model



def get_pretrained_model():
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    )
    pretrained_model.trainable = False
    return pretrained_model


# Initialize model

def create_compiled_custom_model():
    """
    Initialize the Convelutional Neural Network
    """

    base_model = get_pretrained_model()
    flattening_layer = tf.keras.layers.Flatten()
    dense_layer_1 = tf.keras.layers.Dense(250, activation="relu")
    dense_layer_2 = tf.keras.layers.Dense(100, activation="relu")
    dense_layer_3 = tf.keras.layers.Dense(50, activation="relu")
    prediction_layer = tf.keras.layers.Dense(N_LABELS, activation="sigmoid")

    model = tf.keras.models.Sequential(
        [
            base_model,
            flattening_layer,
            dense_layer_1,
            dense_layer_2,
            dense_layer_3,
            prediction_layer,
        ]
    )

    print("\n✅ model initialized")

    return model



def compile_model(optimizer=opt,
                loss=macro_soft_f1,
                metrics=[macro_f1]):
    """
    Compile the Convelutional Neural Network
    """

    print("\n✅ model compiled")

    return model

def train_model(model: model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=64,
                patience=2,
                validation_split=0.3,
                validation_data=None):
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    es = tf.keras.callbacks.EarlyStopping(
                # monitor="val_macro_f1", mode="max", patience=5, verbose=1, restore_best_weights=True
                monitor="val_loss",
                mode="min",
                patience=5,
                verbose=1,
                restore_best_weights=True,
            )

    history = test_model.fit(
                train_ds,
                epochs=EPOCHS,
                validation_data=val_ds,
                verbose=1,
                callbacks=[es]
            )

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history
