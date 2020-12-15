from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
import numpy as np
def diffeq_model(M, t, p):
    m_x, m_y, m_z = M
    dM = [gamma*(m_y*b_z(t)-m_z*b_y(t))-m_x/t2, gamma*(m_z*b_x(t)-m_x*b_z(t))-m_y/t2, gamma*(m_x*b_y(t)-m_y*b_x(t))-(m_z-m_0)/t1]
    return dM
#define parameters
t = np.linspace(0, 10, 100)
gamma=1 #gyromagnetic ratio
b_x = lambda a : 5*math.sin(4*math.pi*a)
b_y = lambda a : 0*math.sin(2*math.pi*a)
b_z = lambda a : 7*math.sin(3*math.pi*a)
t1=1.6
t2=0.5
m_0=1
M0=[0,0,10] #initial conditions
p = [gamma, b_x, b_y, b_z, t1, t2, m_0] #parameter list
M = odeint(diffeq_model, M0, t, args=(p,))
m_array= np.array(M)
#plot mx my mz
plt.plot(t, M[:,0], 'b-', linewidth=2, label= 'M_x')
plt.plot(t, M[:,1], 'r-', linewidth=2, label= 'M_y')
plt.plot(t, M[:,2], 'g-', linewidth=2, label= 'M_z')
plt.xlabel('time')
plt.ylabel('m_x, m_y, m_z')
plt.legend()
plt.show()
