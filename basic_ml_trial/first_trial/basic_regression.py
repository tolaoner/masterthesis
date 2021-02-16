import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from matplotlib import pyplot as plt
#import seaborn as sns
pd.options.display.max_rows = 10
print('imported modules')
train_df = pd.read_csv('generated_data2.csv')
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
#print(feature_columns)
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
def plot_the_loss_curve(iteration, loss):
    """plot loss vs iteration curve"""
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("mean squared error")
    plt.plot(iteration, loss, label="Loss")
    plt.legend()
    plt.ylim([loss.min()*0.95, loss.max()*1.03])
    plt.show()
def create_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    # add feature columns to the model
    model.add(feature_layer)
    # add one linear layer to model to yield a simple linear regressor
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    # construct the layers into a model that tensorflow can execute
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model
def create_deep_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(my_feature_layer)
    model.add(tf.keras.layers.Dense(units = 20,
                                    activation="relu",
                                    name='hidden1'))
    model.add(tf.keras.layers.Dense(units = 12,
                                    activation="relu",
                                    name="hidden2"))
    model.add(tf.keras.layers.Dense(units=1,
                                    name="output"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model
def train_model(model, dataset, iterations, batch_size, label_name):
    """feed a dataset into the model in order to train it."""
    #split the data set into features and label
    features = {name:value for name, value in dataset.items()}
    label = features.pop(label_name)
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=iterations, shuffle=True)
    #get details that will be usefull for plotting the loss curve
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["mean_squared_error"]
    return epochs, rmse
'''my_model = create_model(1, my_feature_layer)
epochs, mse = train_model(my_model, train_df, 50, 5, "Coeff of Exc. Pulse")
plot_the_loss_curve(epochs, mse)
print('finish')'''
my_model = create_deep_model(1, my_feature_layer)
epochs, mse = train_model(my_model, train_df, 50, 15, "Coeff of Exc. Pulse")
print('finish')
