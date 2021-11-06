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
file_path = (base_path / "datasets" / "dc_2vox_tv.csv").resolve()
#true values
data = pd.read_csv(file_path)
label_data = data.iloc[:, 14:56].copy()
feature_data = data.iloc[:, 0:14].copy()
bx_data = data.iloc[:, 14:35].copy()
by_data = data.iloc[:, 35:56].copy()
t_span = np.linspace(0, 0.001, 21)
feature_data_np = np.array(feature_data)
bx_np = np.array(bx_data.iloc[0:100])
by_np = np.array(by_data.iloc[0:100])
#print(bx_np)
#predicted values
model_path = (base_path / "masterthesis" / "deepNN" / "models" / "dc_2vox_tv").resolve()
model = keras.models.load_model(model_path)
predictions = model.predict(feature_data_np[0:100])
#predictions_pd = pd.DataFrame(predictions)
predictions = np.array(predictions)
'''test_data = data.sample(n=2000)
y_test = test_data.iloc[:, 14:56]
x_test = test_data.iloc[:, 0:14]
results = model.evaluate(x_test, y_test, batch_size=40)
print("Test loss, test rmse:", results)'''
#print(predictions)
bx_predictions, by_predictions = np.split(predictions, 2, axis=1)
#print(bx_predictions[:, 0])
#print(bx_np[:, 0])
#plotting
n = np.linspace(1, 100, 100)
plt.figure(1)
plt.title('First Points of True and Predicted B_x')
plt.plot(n, bx_np[:, 0], 'b', ls='-', label="True")
plt.plot(n, bx_predictions[:, 0], 'r', ls='dashed', label='Predicted')
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Data Point (n)')
plt.legend()
plt.show()
plt.figure(2)
plt.title('First Points of True and Predicted B_y')
plt.plot(n, by_np[:, 0], 'b', ls='-', label="True")
plt.plot(n, by_predictions[:, 0], 'r', ls='dashed', label='Predicted')
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Data Point (n)')
plt.legend()
plt.show()
'''
plt.figure(1)
plt.title('True vs Predicted B_x')
plt.plot(t_span, bx_np, 'b', ls='-', label="True B_x")
plt.plot(t_span, bx_predictions[0], 'r', ls='dashed', label="Predicted B_x")
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()

plt.figure(2)
plt.title('True vs Predicted B_y')
plt.plot(t_span, bx_np, 'b', ls='-', label="True B_y")
plt.plot(t_span, by_predictions[0], 'r', ls='dashed', label="Predicted B_y")
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
'''
'''
M_ic_1 = [feature_data.iloc[0]["Mx_IC_1"], feature_data.iloc[0]["My_IC_1"], feature_data.iloc[0]["Mz_IC_1"]]
#print(M_ic_1)
M_true = solve_bloch(t_span, M_ic_1, bx_np, by_np, feature_data.iloc[0]['B_z'])
M_predicted = solve_bloch(t_span, M_ic_1, bx_predictions[0], by_predictions[0], feature_data.iloc[0]['B_z'])
plt.figure(1)
plt.title('Magnetization with Predicted Excitation Pulse')
plt.plot(t_span, M_predicted[:, 0], 'b', label="M_x")
plt.plot(t_span, M_predicted[:, 1], 'r', label="M_y")
plt.plot(t_span, M_predicted[:, 2], 'y', label="M_z")
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
plt.figure(2)
plt.title('Magnetization with True Excitation Pulse')
plt.plot(t_span, M_true[:, 0], 'b', label="M_x")
plt.plot(t_span, M_true[:, 1], 'r', label="M_y")
plt.plot(t_span, M_true[:, 2], 'y', label="M_z")
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()'''