import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from bloch_solver import solve_bloch
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math

base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets" / "2voxel_time_varying.csv").resolve()
data = pd.read_csv(file_path)
label_data = data[['a1', 'a2', 'b1', 'b2']].copy()
feature_data = data.drop(['a1', 'a2', "b1", "b2"], axis=1)
t_span = np.linspace(0, feature_data.iloc[0]['Time'], 150)
a1 = label_data.iloc[0]['a1']
a2 = label_data.iloc[0]['a2']
b1 = label_data.iloc[0]['b1']
b2 = label_data.iloc[0]['b2']
f1 = feature_data.iloc[0]['f1']
f2 = feature_data.iloc[0]['f2']
true_bx = a1*np.cos(f1*t_span) + a2*np.sin(f2*t_span)
true_by = b1*np.cos(f1*t_span) + b2*np.sin(f2*t_span)
#predict
feature_np = np.array(feature_data)
model_path = (base_path / "masterthesis" / "deepNN" / "models" / "2vox_tv2").resolve()
model = keras.models.load_model(model_path)
predictions = model.predict(feature_np[0:1])
predictions_pd = pd.DataFrame(predictions, columns=['a1', 'a2', 'b1', 'b2'])
#print(predictions_pd)
predicted_bx = predictions_pd.iloc[0]['a1']*np.cos(f1*t_span) + predictions_pd.iloc[0]['a2']*np.sin(f2*t_span)
predicted_by = predictions_pd.iloc[0]['b1']*np.cos(f1*t_span) + predictions_pd.iloc[0]['b2']*np.sin(f2*t_span)

file_path2 = (base_path / "datasets" / "2vox_tv_test.csv").resolve()
test_data = pd.read_csv(file_path2)
y_test = data[['a1', 'a2', 'b1', 'b2']].copy()
x_test = data.drop(['a1', 'a2', "b1", "b2"], axis=1)
results = model.evaluate(x_test, y_test, batch_size=40)
print("Test loss, test rmse:", results)

'''plt.figure(1)
plt.title('Predicted vs True Bx')
plt.plot(t_span, true_bx, 'b', ls='-', label='True Bx')
plt.plot(t_span, predicted_bx, 'r', ls='dashed', label='Predicted Bx')
plt.figure(2)
plt.title('Predicted vs True By')
plt.plot(t_span, true_bx, 'b', ls='-', label='True By')
plt.plot(t_span, predicted_bx, 'r', ls='dashed', label='Predicted By')
plt.legend()
plt.show()

M_ic_1 = [feature_data.iloc[0]["Mx_IC_1"], feature_data.iloc[0]["My_IC_1"], feature_data.iloc[0]["Mz_IC_1"]]
M_ic_2 = [feature_data.iloc[0]["Mx_IC_2"], feature_data.iloc[0]["My_IC_2"], feature_data.iloc[0]["Mz_IC_2"]]
#print(M_ic_1)
M_true = solve_bloch(t_span, M_ic_1, true_bx, true_by, feature_data.iloc[0]['B_z_1'])
M_predicted = solve_bloch(t_span, M_ic_1, predicted_bx, predicted_by, feature_data.iloc[0]['B_z_1'])

M_true2 = solve_bloch(t_span, M_ic_2, true_bx, true_by, feature_data.iloc[0]['B_z_2'])
M_predicted2 = solve_bloch(t_span, M_ic_2, predicted_bx, predicted_by, feature_data.iloc[0]['B_z_2'])

plt.figure(1)
plt.title('Voxel 1: Magnetization with True Excitation Pulse ')
plt.plot(t_span, M_true[:, 0], 'b', label="M_x")
plt.plot(t_span, M_true[:, 1], 'r', label="M_y")
plt.plot(t_span, M_true[:, 2], 'y', label="M_z")
plt.legend()

plt.figure(2)
plt.title('Voxel 1: Magnetization with Predicted Excitation Pulse')
plt.plot(t_span, M_predicted[:, 0], 'b', label="M_x")
plt.plot(t_span, M_predicted[:, 1], 'r', label="M_y")
plt.plot(t_span, M_predicted[:, 2], 'y', label="M_z")
plt.legend()

plt.figure(3)
plt.title('Voxel 2: Magnetization with True Excitation Pulse')
plt.plot(t_span, M_true2[:, 0], 'b', label="M_x")
plt.plot(t_span, M_true2[:, 1], 'r', label="M_y")
plt.plot(t_span, M_true2[:, 2], 'y', label="M_z")
plt.legend()

plt.figure(4)
plt.title('Voxel 2: Magnetization with Predicted Excitation Pulse')
plt.plot(t_span, M_predicted2[:, 0], 'b', label="M_x")
plt.plot(t_span, M_predicted2[:, 1], 'r', label="M_y")
plt.plot(t_span, M_predicted2[:, 2], 'y', label="M_z")
plt.legend()
plt.show()'''