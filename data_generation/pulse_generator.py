import numpy as np
import matplotlib.pyplot as plt
t_span = np.linspace(0, 10, 1000)
def generate_pulse():
    f1 = -3
    f2 = 1
    f3 = 2
    a11, a22, a33, b11, b22, b33 = np.random.uniform(0, 1000, size=(6)) #random float values
    a1, a2, a3, b1, b2, b3 = np.random.randint(0, 1000, size=(6)) # random int values
    b_x = lambda t: a1*np.cos(f1*t)+a2*np.cos(f2*t)+a3*np.cos(f3*t)
    b_y = lambda t: b1*np.sin(f1*t)+b2*np.sin(f2*t)+b3*np.sin(f3*t)
    return b_x, b_y

'''b_x, b_y = generate_pulse()
plt.plot(t_span, b_x(t_span), 'r-', linewidth= 2, label='b_x')
plt.plot(t_span, b_y(t_span), 'b-', linewidth= 2, label='b_y')
plt.legend()
plt.show()'''
