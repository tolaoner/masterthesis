from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from pulse_generator import generate_pulse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def solve_bloch(t_span, M_ic, b_x, b_y, b_z):
    """solve bloch equations
        given the time span and parameters (bx, by, bz, t_span, M_ic)"""
    # define parameters
    gamma = 1  # gyromagnetic ratio
    # b_x, b_y = generate_pulse()
    # b_z = lambda a: 10 + 0 * math.sin(200 * math.pi * a)
    t1 = 1.6  # longitudinal relaxation
    t2 = 0.5  # transverse relaxation
    m_0 = 1
    # M_ic = [0, 0, 1]  # initial conditions
    p = [gamma, b_x, b_y, b_z, t1, t2, m_0]  # parameter list
    def diffeq_model(M, t, p):
        m_x, m_y, m_z = M
        t = int(t)
        dM = [gamma * (m_y * b_z - m_z * b_y[t]) - m_x / t2,
              gamma * (m_z * b_x[t] - m_x * b_z) - m_y / t2,
              gamma * (m_x * b_y[t] - m_y * b_x[t]) - (m_z - m_0) / t1]
        return dM
    # solve the diffeq
    m_f = odeint(diffeq_model, M_ic, t_span, args=(p,))
    return m_f
# solve_bloch(t_span, p, M_ic)
t_span = np.linspace(0, 0.001, 21)
M_ic = [0, 0, 1]
b_x, b_y = generate_pulse()
b_z = 10
M = solve_bloch(t_span, M_ic, b_x, b_y, b_z)
#print(len(M[:,1]))
'''base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/dc_timevarying.csv").resolve()
data = pd.read_csv(file_path)
t_span = np.linspace(0, 0.001, 21)
mag_f = data[['Mx_f', 'My_f', 'Mz_f']].copy()
plt.figure()
plt.plot(t_span, M[:, 0], label='Mx')
plt.plot(t_span, M[:, 1], label='My')
plt.plot(t_span, M[:, 2], label='Mz')
plt.xlabel('time')
plt.ylabel('Magnetization')
plt.legend()
plt.show()'''
