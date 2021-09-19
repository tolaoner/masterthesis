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
features = ['Time', 'Mx_IC', 'My_IC', 'Mz_IC', 'Mx_f', 'My_f', 'Mz_f', 'B_z', 'B_x1'
    , 'B_x2', 'B_x3', 'B_x4', 'B_x5', 'B_x6', 'B_x7', 'B_x8', 'B_x9', 'B_x10', 'B_x11', 'B_x12', 'B_x13'
    , 'B_x14', 'B_x15', 'B_x16', 'B_x17', 'B_x18', 'B_x19', 'B_x20', 'B_x21', 'B_y1', 'B_y2', 'B_y3'
    , 'B_y4', 'B_y5', 'B_y6', 'B_y7', 'B_y8', 'B_y9', 'B_y10', 'B_y11', 'B_y12', 'B_y13', 'B_y14', 'B_y15', 'B_y16'
    , 'B_y17', 'B_y18', 'B_y19', 'B_y20', 'B_y21']
# for time varying add---, 'a1', 'a2', "b1", "b2", "f1", 'f2']

list_data = []

for i in range(20000):# data count
    #t = np.random.randint(1, 40) / 1000
    t = 0.001
    t_span = np.linspace(0, 0.001, 21)  # .astype(int)
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
    row_data = [t, mx_ic, my_ic, mz_ic, m_x, m_y, m_z, b_z, b_x[0], b_x[1], b_x[2], b_x[3], b_x[4], b_x[5], b_x[6], b_x[7]
                , b_x[8], b_x[9], b_x[10], b_x[11], b_x[12], b_x[13], b_x[14], b_x[15], b_x[16], b_x[17], b_x[18], b_x[19], b_x[20]
                , b_y[0], b_y[1], b_y[2], b_y[3], b_y[4], b_y[5], b_y[6], b_y[7], b_y[8], b_y[9], b_y[10], b_y[11], b_y[12]
                , b_y[13], b_y[14], b_y[15], b_y[16], b_y[17], b_y[18], b_y[19], b_y[20]]
    # add (, coef_list[0], coef_list[1], coef_list[2], coef_list[3], coef_list[4], coef_list[5]]) for time varying
    list_data.append(row_data)
frame_data = pd.DataFrame(list_data, columns=features)
base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/dc_timevarying.csv").resolve()

with open(file_path, 'a') as f:
    frame_data.to_csv(f, mode='a', index=False, header=not f.tell())
# read_data = pd.read_csv(file_path)
# print(read_data)
print('executed')

# print('Elapsed time = ', time.time()-start_time) # show elapsed time
# plot magnetisation vs time