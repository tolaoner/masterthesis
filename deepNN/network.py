import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from matplotlib import pyplot as plt
pd.options.display.max_rows = 10
print('imported modules')
train_data = pd.read_csv('const_exc_data.csv')
train_df = train_data[:100000].copy()
print(train_df)
feature_columns = []
time = tf.feature_column.numeric_column("Time")
feature_columns.append(time)
m_x = tf.feature_column.numeric_column("Mx_f")
feature_columns.append(m_x)
m_y = tf.feature_column.numeric_column("My_f")
feature_columns.append(m_y)
m_z = tf.feature_column.numeric_column("Mz_f")
feature_columns.append(m_z)
b_z = tf.feature_column.numeric_column("B_z")
feature_columns.append(b_z)
# print(feature_columns)
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


def plot_the_loss_curve(iteration, loss):
    """plot loss vs iteration curve"""
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("mean squared error")
    plt.plot(iteration, loss, label="Loss")
    plt.legend()
    plt.ylim([loss.min()*0.90, loss.max()*1.03])
    plt.show()


def create_deep_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=10,
                                    activation="relu",
                                    name='hidden1'))
    model.add(tf.keras.layers.Dense(units=10,
                                    activation="relu",
                                    name="hidden2"))
    model.add(tf.keras.layers.Dense(units=20,
                                    activation="relu",
                                    name="hidden3"))
    model.add(tf.keras.layers.Dense(units=10,
                                    activation="relu",
                                    name="hidden4"))
    model.add(tf.keras.layers.Dense(units=10,
                                    activation="relu",
                                    name="hidden5"))
    model.add(tf.keras.layers.Dense(units=1,
                                    name="output"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, dataset, iterations, batch_size, label_name):
    """feed a dataset into the model in order to train it."""
    # split the data set into features and label
    features = {name: value for name, value in dataset.items()}
    label = features.pop(label_name)
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=iterations, shuffle=True)
    # get details that will be useful for plotting the loss curve
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["loss"]
    return epochs, rmse

my_model = create_deep_model(0.01, my_feature_layer)
epochs, mse = train_model(my_model, train_data, 50, 40, "B_x")
plot_the_loss_curve(epochs, mse)
#my_model.save('first_model')
print('finish')