from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
import numpy as np

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
        dM = [gamma * (m_y * b_z - m_z * b_y(t)) - m_x / t2,
              gamma * (m_z * b_x(t) - m_x * b_z) - m_y / t2,
              gamma * (m_x * b_y(t) - m_y * b_x(t)) - (m_z - m_0) / t1]
        return dM
    # solve the diffeq
    m_f = odeint(diffeq_model, M_ic, t_span, args=(p,))
    return m_f
# solve_bloch(t_span, p, M_ic)
