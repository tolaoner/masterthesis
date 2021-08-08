from pathlib import Path
from bloch_solver_ti import solve_bloch
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

# start_time = time.time()  # record start time
voxel_count = 5
cnn_features = ['Mx_IC', 'My_IC', 'Mz_IC', 'Mx_f', 'My_f', 'Mz_f', 'B_z']
ff_features = ['Time', 'B_x', 'B_y']
datapoint = []
cnn_data = []
ff_data = []
m_x = []
m_y = []
m_z = []

for i in range(10000):  # data count
    b_x = np.random.uniform(0, 100)
    b_y = np.random.uniform(0, 100)
    t = np.random.randint(1, 40) / 1000
    t_span = np.linspace(0, t, 150)
    # b_x, b_y = generate_pulse(t)
    # add coeff list for time varying exc.-- coeff list is the coefficients of excitation pulse (a1, a2, b1, b2, f1, f2)
    for k in range(voxel_count):
        b_z = np.random.uniform(0, 10)
        mx_ic = 0.2 * np.random.uniform()
        my_ic = 0.2 * np.random.uniform()
        mz_ic = math.sqrt(1 - mx_ic ** 2 - my_ic ** 2)
        M_ic = [mx_ic, my_ic, mz_ic]  # initial conditions
        M = solve_bloch(t_span, M_ic, b_x, b_y, b_z)
        m_x = M[-1, 0]
        m_y = M[-1, 1]
        m_z = M[-1, 2]
        voxel = [mx_ic, my_ic, mz_ic, m_x, m_y, m_z, b_z]
        datapoint.append(voxel)
    cnn_data.append(datapoint)
    datapoint = []
    ff_row = [t, b_x, b_y]
    ff_data.append(ff_row)
cnn_array = np.array(cnn_data)
# print(cnn_array)

# print(cnn_data)
frame_ff = pd.DataFrame(ff_data, columns=ff_features)

# array_cnn = np.array(cnn_data)

base_path = Path(__file__).parent.parent.parent

file_path = (base_path / "datasets/cnn_array").resolve()
np.save(file_path, cnn_array)

file_path_2 = (base_path / "datasets/ff_dataset.csv").resolve()
with open(file_path_2, 'a') as f:
    frame_ff.to_csv(f, mode='a', index=False, header=not f.tell())
print('files are saved')


# returned_data = pd.read_csv(file_path) # read data from csv
# print(returned_data.to_string()) # show data
# print('Elapsed time = ', time.time()-start_time) # show elapsed time
# plot magnetisation vs time

'''plt.plot(t_span1, M[:, 0], 'b-', linewidth=2, label='M_x')
plt.plot(t_span1, M[:, 1], 'r-', linewidth=2, label='M_y')
plt.plot(t_span1, M[:, 2], 'g-', linewidth=2, label='M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()'''