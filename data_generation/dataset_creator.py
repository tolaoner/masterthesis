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
features = ['Time', 'Mx_f', 'My_f', 'Mz_f', 'B_z', 'Coeff of Exc. Pulse']
list_data = []
m_x = []
m_y = []
m_z = []
for i in range(400):
    b_z = np.random.randint(-20, 21, size=(1))
    t = np.random.randint(1, 16) / 1000
    t_span = np.linspace(0, t, 150)
    b_x, b_y, coef_list = generate_pulse() # coeff list is the coefficients of excitation pulse (a1, a2, b1, b2, f1, f2)
    for i in range(1):
        M = solve_bloch(t_span, M_ic, b_x, b_y, b_z[i])
        m_x.append(M[-1, 0])
        m_y.append(M[-1, 1])
        m_z.append(M[-1, 2])
    #print(m_x, '\n', m_y, '\n', m_z)
    row_data=[t, m_x[0], m_y[0], m_z[0], b_z[0], coef_list]
    list_data.append(row_data)
    m_x=[]
    m_y=[]
    m_z=[]
frame_data = pd.DataFrame(list_data, columns=features)
with open('generated_data2.csv', 'a') as f:
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