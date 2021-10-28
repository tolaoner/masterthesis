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
file_path = (base_path / "datasets" / "dc_timevarying.csv").resolve()
#true values
data = pd.read_csv(file_path)
label_data = data.iloc[:, 8:50].copy()
feature_data = data.iloc[:, 0:8].copy()
bx_data = data.iloc[:, 8:29].copy()
by_data = data.iloc[:, 29:50].copy()
t_span = np.linspace(0, 0.001, 21)
feature_data_np = np.array(feature_data)
bx_np = np.array(bx_data.iloc[0])
by_np = np.array(by_data.iloc[0])
#print(bx_np)
#predicted values
model_path = (base_path / "masterthesis" / "deepNN" / "models" / "dc_timevarying").resolve()
model = keras.models.load_model(model_path)
predictions = model.predict(feature_data_np[0:1])
#predictions_pd = pd.DataFrame(predictions)
predictions = np.array(predictions)
test_data = data.sample(n=2000)
y_test = test_data.iloc[:, 8:50]
x_test = test_data.iloc[:, 0:8]
results = model.evaluate(x_test, y_test, batch_size=40)
print("Test loss, test rmse:", results)
#print(predictions)
bx_predictions, by_predictions = np.split(predictions, 2, axis=1)
#print(bx_predictions[0])
#plotting
'''plt.figure(1)
plt.title('True vs Predicted B_x')
plt.plot(t_span, bx_data.iloc[0, :], 'r', ls='dashed', label="True B_x")
plt.plot(t_span, predictions[0, 0:21], 'b', ls='-', label="Predicted B_x")
plt.legend()
plt.show()

plt.figure(2)
plt.title('True vs Predicted B_y')
plt.plot(t_span, by_data.iloc[0, :], 'r', ls='dashed', label="True B_y")
plt.plot(t_span, predictions[0, 21:42], 'b', ls='-', label="Predicted B_y")
plt.legend()
plt.show()
M_ic_1 = [feature_data.iloc[0]["Mx_IC"], feature_data.iloc[0]["My_IC"], feature_data.iloc[0]["Mz_IC"]]
#print(M_ic_1)
#M_true = solve_bloch(t_span, M_ic_1, bx_np, by_np, feature_data.iloc[0]['B_z'])
M_predicted = solve_bloch(t_span, M_ic_1, bx_predictions[0], by_predictions[0], feature_data.iloc[0]['B_z'])
plt.figure(1)
plt.title('Magnetization with Predicted Excitation Pulse')
plt.plot(t_span, M_predicted[:, 0], 'b', label="M_x")
plt.plot(t_span, M_predicted[:, 1], 'r', label="M_y")
plt.plot(t_span, M_predicted[:, 2], 'y', label="M_z")
plt.legend()
plt.show()'''