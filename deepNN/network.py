import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import pathlib as pb
pd.options.display.max_rows = 10
print('imported modules')


base_path = pb.Path(__file__).parent.parent.parent
file_path = (base_path / "datasets" / "2const_exc_data.csv").resolve()
train_data = pd.read_csv(file_path)
label_data = train_data[['B_x', 'B_y']].copy()
feature_data = train_data.drop(['B_x', 'B_y'], axis=1)
# train_df = train_data[:100000].copy()
# print(train_df)


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


def create_deep_model(learning_rate):
    model = Sequential()
    model.add(tf.keras.layers.Dense(units=10,
                                    input_shape=(8,),
                                    activation="relu",
                                    name='hidden1'))
    '''model.add(tf.keras.layers.Dense(units=20,
                                    activation="relu",
                                   name="hidden2"))'''
    model.add(tf.keras.layers.Dense(units=10,
                                    activation="relu",
                                    name="hidden3"))
    '''model.add(tf.keras.layers.Dense(units=20,
                                    activation="relu",
                                    name="hidden4"))'''
    model.add(tf.keras.layers.Dense(units=5,
                                    activation="relu",
                                    name="hidden5"))
    model.add(tf.keras.layers.Dense(units=2,
                                    name="output"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model, feature_set, label_set, iterations, batch_size, label_name):
    """feed a dataset into the model in order to train it."""
    # split the data set into features and label
    label = label_set
    features = feature_set
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=iterations, shuffle=True)
    # get details that will be useful for plotting the loss curve
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["loss"]
    return epochs, rmse


my_model = create_deep_model(0.001)
my_model.summary()
keras.utils.plot_model(my_model, to_file="network_structure_trials/matlab_set/matlab_set.png", show_shapes=True)
epochs, mse = train_model(my_model, feature_data, label_data, 100, 40, ["B_x", "B_y"])
my_model.save('trained_matlab_set')
plot_the_loss_curve(epochs, mse)
print('finish')
