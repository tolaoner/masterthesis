from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
import numpy as np
def diffeq_model(M, t, p):
    m_x, m_y, m_z = M
    dM = [gamma*(m_y*b_z-m_z*b_y)-m_x/t2, gamma*(m_z*b_x-m_x*b_z)-m_y/t2, gamma*(m_x*b_y-m_y*b_x)-(m_z-m_0)/t1]
    return dM
#define parameters
t = np.linspace(0, 10, 10)
gamma=1 #gyromagnetic ratio
b_x = math.sin(t)
b_y = 2*math.cos(t)
b_z=1
t1=5
t2=10
m_0=10
M0=[0,0,10] #initial conditions
p = [gamma, b_x, b_y, b_z, t1, t2, m_0] #parameter list
M = odeint(diffeq_model, M0, t, args=(p,))
m_array= np.array(M)
#plot mx my mz
plt.plot(t, M[:,0], 'r-', linewidth=2, label= 'M_x')
plt.plot(t, M[:,1], 'g-', linewidth=2, label= 'M_y')
plt.plot(t, M[:,2], 'b-', linewidth=2, label= 'M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()
