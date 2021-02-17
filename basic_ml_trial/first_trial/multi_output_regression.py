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
train_data = pd.read_csv("generated_data3.csv")
train_label = np.array(train_data[['a1', 'a2', 'b1', 'b2', 'f1', 'f2']].copy())
train_data = train_data.drop(['a1', 'a2', 'b1', 'b2', 'f1', 'f2'], axis=1)
def train_model(model, feature_data, label_data, batch_size, epochs):
    history = model.fit(feature_data, label_data,
                        batch_size=batch_size,
                        epochs=epochs)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist['loss']
    acc = hist['accuracy']
    return epochs, mse, acc
#create model
learning_rate=1
my_model = multi_output_model(learning_rate)
#define parameters
batch = 15
epochs = 50
#train the model
epochs, mse, acc = train_model(my_model, train_data, train_label, batch, epochs)
#plot the loss vs epoch graph
plt.figure()
plt.ylabel('Loss&Accuracy')
plt.xlabel('Epochs')
plt.plot(epochs, mse, label="Loss vs Epochs")
plt.plot(epochs, acc, label="Accuaracy vs Epochs")
plt.legend()
plt.show()