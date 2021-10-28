from pathlib import Path
from bloch_solver import solve_bloch
import numpy as np
import math
from pulse_generator import generate_pulse
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

# start_time = time.time()  # record start time
voxel_count = 2
features = ['Time', 'Mx_IC_1', 'My_IC_1', 'Mz_IC_1', 'Mx_f_1', 'My_f_1', 'Mz_f_1', 'Mx_IC_2', 'My_IC_2', 'Mz_IC_2',
            'Mx_f_2', 'My_f_2', 'Mz_f_2', 'B_z_1', 'B_z_2', 'a1', 'a2', "b1", "b2", "f1", 'f2']
list_data = []
m_x = []
m_y = []
m_z = []
mx_ic = []
my_ic = []
mz_ic = []
b_z = []
for i in range(1000):  # data count
    # b_x = np.random.uniform(0, 100)
    # b_y = np.random.uniform(0, 100)
    t = np.random.randint(1, 40) / 1000
    t_span = np.linspace(0, t, 150)
    # b_x, b_y = generate_pulse(t)
    b_x, b_y, coeff_list = generate_pulse() # add coeff list for time varying exc.
    # -- coeff list is the coefficients of excitation pulse (a1, a2, b1, b2, f1, f2)
    for k in range(voxel_count):
        b_z.append(np.random.uniform(-20, 21))
        mx_ic.append(0.2 * np.random.uniform())
        my_ic.append(0.2 * np.random.uniform())
        mz_ic.append(math.sqrt(1 - mx_ic[k] ** 2 - my_ic[k] ** 2))
        M_ic = [mx_ic[k], my_ic[k], mz_ic[k]]  # initial conditions
        M = solve_bloch(t_span, M_ic, b_x, b_y, b_z[k])
        m_x.append(M[-1, 0])
        m_y.append(M[-1, 1])
        m_z.append(M[-1, 2])
    # print(m_x, '\n', m_y, '\n', m_z)
    row_data = [t, mx_ic[0], my_ic[0], mz_ic[0], m_x[0], m_y[0], m_z[0], mx_ic[1], my_ic[1], mz_ic[1],
                m_x[1], m_y[1], m_z[1], b_z[0], b_z[1], coeff_list[0], coeff_list[1], coeff_list[2], coeff_list[3], coeff_list[4], coeff_list[5]]
    list_data.append(row_data)
    m_x = []
    m_y = []
    m_z = []
    mx_ic = []
    my_ic = []
    mz_ic = []
frame_data = pd.DataFrame(list_data, columns=features)
base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/2vox_tv_test.csv").resolve()
with open(file_path, 'a') as f:
    frame_data.to_csv(f, mode='a', index=False, header=not f.tell())

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