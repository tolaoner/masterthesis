import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from pathlib import Path
import plot_const_pulse as pcp
from matplotlib import pyplot as plt
base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/cnn_array.npy").resolve()
file_path2 = (base_path / "datasets/ff_dataset.csv").resolve()

ff_data = pd.read_csv(file_path2)
time_data = ff_data['Time'].copy()
time_data = time_data.to_numpy()
#print(time_data[0])
#t_span = np.linspace(0, time_data[0], 150)

cnn_data = np.load(file_path)
cnn_sample = cnn_data[0:2]
cnn_sample = cnn_sample.reshape(2, 5, 7, 1)
#print(cnn_sample[0])
#print(cnn_sample[0].shape)

model_path = (base_path / "masterthesis" / "CNN" / "models" / "const_multivox_cnn").resolve()
model = keras.models.load_model(model_path)

predictions = model.predict(cnn_sample[0], time_data[0])
predictions_pd = pd.DataFrame(predictions, columns=['B_x', 'B_y'])
print(predictions_pd)