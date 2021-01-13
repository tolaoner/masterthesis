from bloch_solver import solve_bloch
from pulse_generator import generate_pulse
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
t_span1 = np.linspace(0, 0.016, 160)
t_span2 = np.linspace(0, 10, 20000)
M_ic = [0, 0, 1]  # initial conditions
M = solve_bloch(t_span1, M_ic)
# plot magnetisation vs time
plt.plot(t_span1, M[:, 0], 'b-', linewidth=2, label='M_x')
#plt.plot(t_span2, M[:, 1], 'r-', linewidth=2, label='M_y')
#plt.plot(t_span2, M[:, 2], 'g-', linewidth=2, label='M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()