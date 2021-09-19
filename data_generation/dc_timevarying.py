from pathlib import Path
from bloch_solver import solve_bloch
from pulse_generator import generate_pulse
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

# start_time = time.time()  # record start time
pd.options.display.max_rows = 10
pd.options.display.max_columns = 20
voxel_count = 1
M_ic = [0, 0, 1]  # initial conditions
features = ['Time', 'Mx_IC', 'My_IC', 'Mz_IC', 'Mx_f', 'My_f', 'Mz_f', 'B_z', 'B_x', 'B_y']
# for time varying add---, 'a1', 'a2', "b1", "b2", "f1", 'f2']

list_data = []

for i in range(10000):# data count
    #t = np.random.randint(1, 40) / 1000
    t = 0.001
    t_span = np.linspace(0, 20, 21).astype(int)
    b_x, b_y = generate_pulse()
    # add coeff list for time varying exc.-- coeff list is the coefficients of excitation pulse (a1, a2, b1, b2, f1, f2)
    for k in range(voxel_count):
        b_z = np.random.uniform(-20, 20)
        mx_ic = 0.2 * np.random.uniform()
        my_ic = 0.2 * np.random.uniform()
        mz_ic = math.sqrt(1 - mx_ic ** 2 - my_ic ** 2)
        M = solve_bloch(t_span, M_ic, b_x, b_y, b_z)
        m_x = M[-1, 0]
        m_y = M[-1, 1]
        m_z = M[-1, 2]
    # print(m_x, '\n', m_y, '\n', m_z)
    row_data = [t, mx_ic, my_ic, mz_ic, m_x, m_y, m_z, b_z, b_x, b_y]
    # add (, coef_list[0], coef_list[1], coef_list[2], coef_list[3], coef_list[4], coef_list[5]]) for time varying
    list_data.append(row_data)

frame_data = pd.DataFrame(list_data, columns=features)
print(frame_data['B_z'])
base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/dc_timevarying.csv").resolve()

with open(file_path, 'a') as f:
    frame_data.to_csv(f, mode='a', index=False, header=not f.tell())
# read_data = pd.read_csv(file_path)
# print(read_data)
print('executed')

# print('Elapsed time = ', time.time()-start_time) # show elapsed time
# plot magnetisation vs time