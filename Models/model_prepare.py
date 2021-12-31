from Experiments.settings import *
from Models.ResNet.Resnet import resnet
from Losses.Losses import get_loss_function
import tensorflow as tf


def prepare_model(model_type: Models,
                  approach: Approaches,
                  num_classes: int,
                  input_shape: list,
                  save_path="",
                  optimizer=None,
                  saved_weights=None):

    model = resnet(input_shape=input_shape, num_classes=num_classes, model_type=model_type)

    # Loss function
    loss, callbacks = get_loss_function(approach=approach)

    # Optimizer
    optimizer = optimizer if optimizer is not None \
        else tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss=loss)

    return model, callbacks
