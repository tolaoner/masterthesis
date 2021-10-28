import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from pathlib import Path
import plot_const_pulse as pcp
from matplotlib import pyplot as plt
from bloch_solver_ti import solve_bloch
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math

base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/cnn_array.npy").resolve()
file_path2 = (base_path / "datasets/ff_dataset.csv").resolve()

ff_data = pd.read_csv(file_path2)
time_data = ff_data['Time'].copy()
t_span = np.linspace(0, time_data[0], 150)
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

predictions = model([cnn_sample, time_data[0:2]], training=False)
predictions_pd = pd.DataFrame(predictions.numpy(), columns=['B_x', 'B_y'])
#print(predictions_pd)
evaluation_data = ff_data[['B_x','B_y']].copy()
results = model.evaluate([cnn_data[0:6000], time_data[0:6000]], evaluation_data[0:6000], batch_size=40)
print("Test loss, test rmse:", results)
#pcp.plot_const_evol(ff_data, predictions_pd)
'''
cnn_pd = pd.DataFrame(cnn_data[0], columns=['Mx_IC', 'My_IC', 'Mz_IC', 'Mx_f', 'My_f', 'Mz_f', 'B_z'])
M_ic_1 = [cnn_pd.iloc[0]['Mx_IC'], cnn_pd.iloc[0]['My_IC'], cnn_pd.iloc[0]['Mz_IC']]

M_true = solve_bloch(t_span, M_ic_1, ff_data.iloc[0]['B_x'], ff_data.iloc[0]['B_y'], cnn_pd.iloc[0]['B_z'])
M_predicted = solve_bloch(t_span, M_ic_1, predictions_pd.iloc[0]['B_x'], predictions_pd.iloc[0]['B_y'], cnn_pd.iloc[0]['B_z'])

plt.figure(1)
plt.title('Magnetization with True Excitation Pulse')
plt.plot(t_span, M_true[:, 0], 'b', label="M_x")
plt.plot(t_span, M_true[:, 1], 'r', label="M_y")
plt.plot(t_span, M_true[:, 2], 'y', label="M_z")
plt.legend()

plt.figure(2)
plt.title('Magnetization with Predicted Excitation Pulse')
plt.plot(t_span, M_predicted[:, 0], 'b', label="M_x")
plt.plot(t_span, M_predicted[:, 1], 'r', label="M_y")
plt.plot(t_span, M_predicted[:, 2], 'y', label="M_z")
plt.legend()
plt.show()'''