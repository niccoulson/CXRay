
from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from typing import Tuple

# Initialize model

def initialize_model(X: np.ndarray) -> Model:
    """
    Initialize the Convelutional Neural Network
    """

    model = Sequential()

    print("\n✅ model initialized")

    return model



def compile_model(model: Model, learning_rate: float) -> Model:
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
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history
