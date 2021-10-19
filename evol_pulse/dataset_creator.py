from pathlib import Path
from bloch_solver_ti import solve_bloch
#from pulse_generator import generate_pulse
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

# start_time = time.time()  # record start time

voxel_count = 1
features = ['Time', 'Mx_IC', 'My_IC', 'Mz_IC', 'Mx_f', 'My_f', 'Mz_f', 'B_z', 'B_x', 'B_y']
# for time varying add---, 'a1', 'a2', "b1", "b2", "f1", 'f2']

list_data = []

for i in range(6000):  # data count
    b_x = np.random.uniform(0, 100)
    b_y = 0  # np.random.uniform(0, 100)
    t = np.random.randint(1, 40) / 1000
    t_span = np.linspace(0, t, 150)
    # b_x, b_y = generate_pulse(t)
    # add coeff list for time varying exc.-- coeff list is the coefficients of excitation pulse (a1, a2, b1, b2, f1, f2)
    for k in range(voxel_count):
        M_ic = []  # initial conditions
        b_z = np.random.uniform(0, 10)
        mx_ic = 0.2 * np.random.uniform()
        my_ic = 0.2 * np.random.uniform()
        mz_ic = math.sqrt(1 - mx_ic ** 2 - my_ic ** 2)
        M_ic = [mx_ic, my_ic, mz_ic]
        M = solve_bloch(t_span, M_ic, b_x, b_y, b_z)
        m_x = M[-1, 0]
        m_y = M[-1, 1]
        m_z = M[-1, 2]
    # print(m_x, '\n', m_y, '\n', m_z)
    row_data = [t, mx_ic, my_ic, mz_ic, m_x, m_y, m_z, b_z, b_x, b_y]
    # add (, coef_list[0], coef_list[1], coef_list[2], coef_list[3], coef_list[4], coef_list[5]]) for time varying
    list_data.append(row_data)

frame_data = pd.DataFrame(list_data, columns=features)

base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/test_by0.csv").resolve()

with open(file_path, 'a') as f:
    frame_data.to_csv(f, mode='a', index=False, header=not f.tell())
print('executed')

# print('Elapsed time = ', time.time()-start_time) # show elapsed time
# plot magnetisation vs time

'''plt.plot(t_span1, M[:, 0], 'b-', linewidth=2, label='M_x')
plt.plot(t_span1, M[:, 1], 'r-', linewidth=2, label='M_y')
plt.plot(t_span1, M[:, 2], 'g-', linewidth=2, label='M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()'''