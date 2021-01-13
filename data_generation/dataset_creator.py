from bloch_solver import solve_bloch
from pulse_generator import generate_pulse
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
b_x, b_y = generate_pulse()
t_span1 = np.linspace(0, 0.016, 160)
t_span2 = np.linspace(0, 10, 20000)
gamma = 1  # gyromagnetic ratio
b_z = lambda a: 10+0*np.sin(200*math.pi*a)
t1 = 1.6  # longitudinal relaxation
t2 = 0.5  # transverse relaxation
m_0 = 1
M_ic = [0, 0, 1]  # initial conditions
p = [gamma, b_x, b_y, b_z, t1, t2, m_0] #parameter list
M = solve_bloch(t_span1, p, M_ic)
# plot magnetisation vs time
plt.plot(t_span1, M[:, 0], 'b-', linewidth=2, label='M_x')
plt.plot(t_span1, M[:, 1], 'r-', linewidth=2, label='M_y')
plt.plot(t_span1, M[:, 2], 'g-', linewidth=2, label='M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()