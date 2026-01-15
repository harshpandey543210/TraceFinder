"""
Hybrid CNN Model for Scanner Detection (TraceFinder)
Combines residual images with handcrafted features (PRNU correlation + enhanced features).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def build_hybrid_model(
    img_shape: Tuple[int, int, int],
    feat_shape: Tuple[int],
    num_classes: int
) -> keras.Model:
    """
    Builds the Hybrid CNN model for scanner fingerprint detection.

    Args:
        img_shape: Tuple (H, W, C), e.g. (256, 256, 1) for grayscale residuals
        feat_shape: Tuple (num_features,), e.g. (17,) for PRNU + enhanced features
        num_classes: Int, number of scanner classes

    Returns:
        keras.Model: Compiled hybrid model ready for training
    """
    
    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    img_in = keras.Input(shape=img_shape, name="residual")
    feat_in = keras.Input(shape=feat_shape, name="handcrafted")

    # ------------------------------------------------------------------
    # Image Branch (CNN) - High-Pass Filter + Feature Extraction
    # ------------------------------------------------------------------
    # Laplacian High-Pass kernel (fixed, non-trainable)
    hp_kernel = np.array(
        [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    ).reshape((3, 3, 1, 1))

    # Apply High-Pass Filter
    hp = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        trainable=False,
        name="hp_filter",
    )(img_in)

    # Convolutional Blocks - Progressive Feature Extraction
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(hp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)  # Spatial pooling -> (128,)

    # ------------------------------------------------------------------
    # Features Branch (Dense) - Handcrafted Features
    # ------------------------------------------------------------------
    f = layers.Dense(64, activation="relu")(feat_in)
    f = layers.Dropout(0.2)(f)  # (num_features,) -> (64,)

    # ------------------------------------------------------------------
    # Fusion & Classification Head
    # ------------------------------------------------------------------
    z = layers.Concatenate()([x, f])  # (128 + 64 = 192,)
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Dropout(0.4)(z)

    out = layers.Dense(num_classes, activation="softmax", name="scanner_logits")(z)

    # ------------------------------------------------------------------
    # Create & Initialize Model
    # ------------------------------------------------------------------
    model = keras.Model(
        inputs=[img_in, feat_in],
        outputs=out,
        name="Hybrid_Scanner_CNN"
    )

    # Lock High-Pass filter weights (non-trainable)
    model.get_layer("hp_filter").set_weights([hp_kernel])

    return model
