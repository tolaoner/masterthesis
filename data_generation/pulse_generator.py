import numpy as np
import matplotlib.pyplot as plt
#t_span = np.linspace(0, 10, 1000)
def generate_pulse():
    """generate random pulse in the form of sinusoids and cosinusoids"""
    f1, f2 = np.random.randint(-3, 4, size=(2))#frequency values
    a11, a22, a33, b11, b22, b33 = np.random.uniform(0, 1000, size=(6)) #random float values
    a1, a2, b1, b2 = np.random.randint(100, 1000, size=(4)) # random int values
    coef_list = [a1, a2, b1, b2, f1, f2]
    b_x = lambda t: a1*np.cos(f1*t)+a2*np.cos(f2*t)
    b_y = lambda t: b1*np.sin(f1*t)+b2*np.sin(f2*t)
    return b_x, b_y, coef_list
'''t_span = np.linspace(0, 10, 160)
b_x, b_y = generate_pulse()
plt.plot(t_span, b_x(t_span), 'r-', linewidth= 2, label='b_x')
plt.plot(t_span, b_y(t_span), 'b-', linewidth= 2, label='b_y')
plt.legend()
plt.show()'''
