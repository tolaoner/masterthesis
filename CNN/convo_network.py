import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
import pandas as pd
import numpy as np
from matplotlib import pyplot as pl
from pathlib import Path
print('imported modules')

base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/cnn_array.npy").resolve()
file_path2 = (base_path / "datasets/ff_dataset.csv").resolve()

cnn_data = np.load(file_path)
cnn_data = cnn_data.reshape(10000, 5, 7, 1)

label_data = pd.read_csv(file_path2)
time_data = label_data['Time'].copy()
time_data = time_data.to_numpy()
label_data = label_data.drop(['Time'], axis=1)
print(time_data)

# print(cnn_data)

visible = Input(shape=(5, 7, 1))
x = Conv2D(32, (2, 2))(visible)
x = Activation("relu")(x)
x = Conv2D(64, (2, 2))(x)
x = Activation("relu")(x)
x = Conv2D(32, (2, 2))(x)
x = Activation("relu")(x)
x = Flatten()(x)
'''x = Dense(64)(x)
x = Activation("relu")(x)'''
x = Dense(32)(x)
x = Activation("relu")(x)
x = Dense(16, activation="relu")(x)
visible2 = Input(shape=(1))
z = Dense(10, activation='relu')(visible2)
z = Dense(16, activation='relu')(z)
combined = concatenate([x, z])
y = Dense(20, activation="relu")(combined)
y = Dense(10, activation="relu")(y)
output = Dense(2, activation="relu")(y)

model = Model(inputs=[visible, visible2], outputs=output)
opt = tf.keras.optimizers.Adam(lr=0.002)
model.compile(optimizer=opt, loss='mean_squared_error',
              metrics=[RootMeanSquaredError()])
history = model.fit([cnn_data, time_data], label_data, epochs=150)
model.save('models/const_multivox_cnn')


'''learning_rate = 0.001

model = Sequential()
model.add(keras.layers.Conv2D(32, (1, 1), activation='relu', input_shape=(32, 5, 7, 1)))
model.add(keras.layers.Conv2D(64, (1, 1), activation='relu'))
model.add(keras.layers.Conv2D(64, (1, 1), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
combined = tf.concat(model.layers[5].output, time_data)

model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(2))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(cnn_data, label_data, epochs=70)'''


