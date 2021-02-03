from bloch_solver import solve_bloch
from pulse_generator import generate_pulse
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
'''t_span1 = np.linspace(0, 0.016, 160)
t_span2 = np.linspace(0, 10, 20000)'''
M_ic = [0, 0, 1]  # initial conditions
b_z = [-20, 10, 0, 10, 20]
features = ['M_ic', 'T', 'Mx_f', 'My_f', 'Mz_f', 'B_z']
list_data = []
m_x = []
m_y = []
m_z = []
for i in range(10):
    for i in range(5):
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
#pd.set_option('display.max_columns',None)
#print(frame_data.iloc[1]['Mx_f'])
with open('generated_data.csv', 'a') as f:
    frame_data.to_csv(f, mode='a', index=False, header=not f.tell())
returned_data = pd.read_csv('generated_data.csv')
#print(returned_data.to_string())
# plot magnetisation vs time
'''plt.plot(t_span1, M[:, 0], 'b-', linewidth=2, label='M_x')
plt.plot(t_span1, M[:, 1], 'r-', linewidth=2, label='M_y')
plt.plot(t_span1, M[:, 2], 'g-', linewidth=2, label='M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()'''