import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
def plot_const_evol_by0(true_exc_data, predicted_data):
    true_exc_pulse = true_exc_data[['Time', 'B_z', 'B_x', 'B_y']].copy()
    plt.figure()
    #plt.figure(1)
    #plt.subplot(211)
    plt.axhline(y=true_exc_pulse['B_x'][0], color='b', ls='-', linewidth=2, label='True B_x')
    plt.axhline(y=predicted_data['B_x'][0], color='r', ls='dashed', linewidth=2, label='Predicted B_x')
    # plt.axhline(y=true_exc_pulse['B_y'][0], color='r', ls='-', linewidth=2, label='B_y')
    # plt.axhline(y=true_exc_pulse['B_z'][0], color='g', ls='-', linewidth=2, label='B_z')
    plt.xlim(right=true_exc_pulse['Time'][0])
    plt.xlabel('Time')
    plt.ylabel('Excitation Pulse')
    plt.title('True Excitation Pulse')
    plt.ylim([true_exc_pulse['B_x'][0].min() * 0.99, true_exc_pulse['B_x'][0].max() * 1.01])
    plt.legend()
    for var in (true_exc_pulse['B_x'][0], predicted_data['B_x'][0]):
        plt.annotate('%0.2f' % var.max(), xy=(1, var.max()), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    '''plt.subplot(212)
    plt.axhline(y=predicted_data['B_x'][0], color='b', ls='-', linewidth=2, label='B_x')
    #plt.axhline(y=predicted_data['B_y'][0], color='r', ls='-', linewidth=2, label='B_y')
    plt.xlim(right=true_exc_pulse['Time'][0])
    plt.xlabel('Time')
    plt.ylabel('Excitation Pulse')
    plt.title('Predicted Excitation Pulse')'''

    #plt.tight_layout()
    plt.show()
def plot_const_evol(true_exc_data, predicted_data):
    true_exc_pulse = true_exc_data[['Time', 'B_z', 'B_x', 'B_y']].copy()
    #plt.figure()
    #plt.figure(1)
    plt.subplot(211)
    plt.tight_layout()
    plt.axhline(y=true_exc_pulse['B_x'][0], color='b', ls='-', linewidth=2, label='True B_x')
    plt.axhline(y=predicted_data['B_x'][0], color='r', ls='dashed', linewidth=2, label='Predicted B_x')
    # plt.axhline(y=true_exc_pulse['B_y'][0], color='r', ls='-', linewidth=2, label='B_y')
    # plt.axhline(y=true_exc_pulse['B_y'][0], color='g', ls='-', linewidth=2, label='B_z')
    plt.xlim(right=true_exc_pulse['Time'][0])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('True vs Predicted B_x')
    #plt.ylim([true_exc_pulse['B_x'][0].min() * 0.9, true_exc_pulse['B_x'][0].max() * 1.1])
    for var in (true_exc_pulse['B_x'][0], predicted_data['B_x'][0]):
        plt.annotate('%0.2f' % var.max(), xy=(1, var.max()), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.legend()

    plt.subplot(212)
    plt.tight_layout()
    plt.axhline(y=true_exc_data['B_y'][0], color='b', ls='-', linewidth=2, label='True B_y')
    plt.axhline(y=predicted_data['B_y'][0], color='r', ls='dashed', linewidth=2, label='Predicted B_y')
    plt.xlim(right=true_exc_pulse['Time'][0])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('True vs Predicted B_y')
    #plt.ylim([true_exc_pulse['B_y'][0].min() * 0.9, true_exc_pulse['B_y'][0].max() * 1.1])
    plt.legend()
    for var in (true_exc_pulse['B_x'][0], predicted_data['B_x'][0], true_exc_data['B_y'][0], predicted_data['B_y'][0]):
        plt.annotate('%0.2f' % var.max(), xy=(1, var.max()), xytext=(8, 0),
        xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.show()