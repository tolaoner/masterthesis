import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
pd.options.display.max_rows = 10
def multi_output_model(learning_rate):
    inputs = keras.Input(shape=(5,))
    dense = keras.layers.Dense(10, activation="relu")
    x = dense(inputs)
    x = keras.layers.Dense(8, activation="relu")(x)
    outputs = keras.layers.Dense(6)(x)
    model = keras.Model(inputs=inputs,
                        outputs=outputs,
                        name="multi_output_model")
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  metrics=["accuracy"])
    return model
train_data = pd.read_csv("generated_data2.csv")
train_label = train_data.pop("Coeff of Exc. Pulse")
def train_model(model, feature_data, label_data, batch_size, epochs):
    history = model.fit(feature_data, label_data,
                        batch_size=batch_size,
                        epochs=epochs)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["mean_squared_error"]
    return epochs, rmse
my_model = multi_output_model(0.01)
epochs, mse = train_model(my_model, train_data, train_label, 10, 20)
