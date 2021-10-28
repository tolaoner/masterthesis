import tensorflow as tf
from tensorflow import keras
from bloch_solver_ti import solve_bloch
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets" / "const_1vox" / "1const_exc_data.csv").resolve()
file_path2 = (base_path / "datasets" / "const_1vox" / "norm_1const_exc.csv").resolve()

test_data = pd.read_csv(file_path)
t_span = np.linspace(0, test_data.iloc[0]['Time'], 150)
#M_ic = [test_data.iloc[0]['Mx_IC'], test_data.iloc[0]['My_IC'],test_data.iloc[0]['Mz_IC']] this is for one voxel
M_ic_1 = [test_data.iloc[0]['Mx_IC'], test_data.iloc[0]['My_IC'], test_data.iloc[0]['Mz_IC']]
#M_ic_2 = [test_data.iloc[1]['Mx_IC_2'], test_data.iloc[1]['My_IC_2'], test_data.iloc[1]['Mz_IC_2']]
#print(test_data.iloc[0]['B_x'])

# predict bx-by with normalized data
predict_data = pd.read_csv(file_path2)
#print(predict_data.iloc[0]['B_x'])
test_features = predict_data.drop(['B_x', 'B_y'], axis=1)
test_features_np = np.array(test_features)
model_path = (base_path / "masterthesis" / "deepNN" / "models" / "600k_norm_1voxel").resolve()
model = keras.models.load_model(model_path)
predictions = model.predict(test_features_np[0:1])
predictions_pd = pd.DataFrame(predictions, columns=[['B_x', 'B_y']])
#print(test_features_np[1:2])
#print(test_features_np)

M_true = solve_bloch(t_span, M_ic_1, test_data.iloc[0]['B_x'], test_data.iloc[0]['B_y'], test_data.iloc[0]['B_z'])
M_predicted = solve_bloch(t_span, M_ic_1, predictions_pd.iloc[0]['B_x'], predictions_pd.iloc[0]['B_y'], test_data.iloc[0]['B_z'])

#M_true2 = solve_bloch(t_span, M_ic_2, test_data.iloc[1]['B_x'], test_data.iloc[1]['B_y'], test_data.iloc[1]['B_z'])
#M_predicted2 = solve_bloch(t_span, M_ic_2, predictions_pd.iloc[0]['B_x'], predictions_pd.iloc[0]['B_y'], test_data.iloc[1]['B_z'])

plt.figure(1)
plt.title('Magnetization with True Excitation Pulse')
plt.plot(t_span, M_true[:, 0], 'b', label="M_x")
plt.plot(t_span, M_true[:, 1], 'r', label="M_y")
plt.plot(t_span, M_true[:, 2], 'y', label="M_z")
plt.legend()
plt.show()

plt.figure(2)
plt.title('Magnetization with Predicted Excitation Pulse')
plt.plot(t_span, M_predicted[:, 0], 'b', label="M_x")
plt.plot(t_span, M_predicted[:, 1], 'r', label="M_y")
plt.plot(t_span, M_predicted[:, 2], 'y', label="M_z")
plt.legend()
plt.show()

'''plt.figure(3)
plt.title('Magnetization with True Excitation Pulse')
plt.plot(t_span, M_true2[:, 0], 'b', label="M_x")
plt.plot(t_span, M_true2[:, 1], 'r', label="M_y")
plt.plot(t_span, M_true2[:, 2], 'y', label="M_z")
plt.legend()
plt.show()

plt.figure(4)
plt.title('Magnetization with Predicted Excitation Pulse')
plt.plot(t_span, M_predicted2[:, 0], 'b', label="M_x")
plt.plot(t_span, M_predicted2[:, 1], 'r', label="M_y")
plt.plot(t_span, M_predicted2[:, 2], 'y', label="M_z")
plt.legend()
plt.show()'''