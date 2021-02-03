from bloch_solver import solve_bloch
from pulse_generator import generate_pulse
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
#start_time = time.time()  # record start time
voxel_count = 2
M_ic = [0, 0, 1]  # initial conditions
features = ['M_ic', 'T', 'Mx_f', 'My_f', 'Mz_f', 'B_z']
list_data = []
m_x = []
m_y = []
m_z = []
for i in range(1):
    for i in range(voxel_count):
        b_z = np.random.randint(-20, 21, size=(2))
        t = np.random.randint(1, 16) / 100
        t_span = np.linspace(0, t, 150)
        b_x, b_y = generate_pulse()
        M = solve_bloch(t_span, M_ic, b_x, b_y, b_z[i])
        m_x.append(M[-1, 0])
        m_y.append(M[-1, 1])
        m_z.append(M[-1, 2])
    #print(m_x, '\n', m_y, '\n', m_z)
    row_data=[M_ic, t, m_x, m_y, m_z, b_z]
    list_data.append(row_data)
    m_x=[]
    m_y=[]
    m_z=[]
frame_data = pd.DataFrame(list_data, columns=features)
with open('generated_data.csv', 'a') as f:
    frame_data.to_csv(f, mode='a', index=False, header=not f.tell())

#returned_data = pd.read_csv('generated_data.csv') # read data from csv
#print(returned_data.to_string()) # show data
#print('Elapsed time = ', time.time()-start_time) # show elapsed time
# plot magnetisation vs time
'''plt.plot(t_span1, M[:, 0], 'b-', linewidth=2, label='M_x')
plt.plot(t_span1, M[:, 1], 'r-', linewidth=2, label='M_y')
plt.plot(t_span1, M[:, 2], 'g-', linewidth=2, label='M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()'''