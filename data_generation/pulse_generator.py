import numpy as np
import math
import matplotlib.pyplot as plt
from plot_const_pulse import plot_const_evol as evol
#t_span = np.linspace(0, 10, 1000)
def generate_pulse():
    """generate random pulse"""
    # frequency values
    f1, f2 = np.random.randint(5000, 10000, size=(2))
    # random float values
    # a11, a22, a33, b11, b22, b33 = np.random.uniform(0, 1000, size=(6))
    # random int values
    a1, a2, b1, b2 = np.random.randint(100, 1000, size=(4))
    # print(a1, a2, b1, b2, f1, f2)
    coef_list = [a1, a2, b1, b2, f1, f2]
    '''above code is for time-varying excitation'''
    # limit_value = math.pi/(2*time)
    # b_x = np.random.uniform(-limit_value, limit_value)
    # b_y = 0 # np.random.randint(-1000,1000)
    #b_x = lambda t: a1*np.cos(f1*t)+a2*np.cos(f2*t)
    #b_y = lambda t: b1*np.sin(f1*t)+b2*np.sin(f2*t)
    b_x = np.random.uniform(0, 500, size=(21))
    b_y = np.random.uniform(0, 200, size=(21))
    return b_x, b_y
#a = [c for c in range(21)]
'''a = np.linspace(0, 20, 21)
print(a)
b_x, b_y = generate_pulse()
print(b_x[0])
t_span = np.linspace(0, 0.001, 21)
print(b_x)
print(b_y)
plt.plot(t_span, b_x, 'b-', linewidth=2, label='b_x')
plt.plot(t_span, b_y, 'r-', linewidth=2, label='b_y')
plt.xlabel('time')
plt.ylabel('b_x, b_y')
plt.legend()
plt.show()'''
'''t_span = np.linspace(0, 10, 160)
plt.plot(t_span, b_x(t_span), 'r-', linewidth= 2, label='b_x')
plt.plot(t_span, b_y(t_span), 'b-', linewidth= 2, label='b_y')
plt.legend()
plt.show()'''
