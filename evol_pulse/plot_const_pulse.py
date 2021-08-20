import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
def plot_const_evol(true_exc_data, predicted_data):
    true_exc_pulse = true_exc_data[['Time', 'B_z', 'B_x', 'B_y']].copy()

    plt.figure(1)
    plt.subplot(211)
    plt.axhline(y=true_exc_pulse['B_x'][0], color='b', ls='-', linewidth=2, label='B_x')
    plt.axhline(y=true_exc_pulse['B_y'][0], color='r', ls='-', linewidth=2, label='B_y')
    plt.axhline(y=true_exc_pulse['B_z'][0], color='g', ls='-', linewidth=2, label='B_z')
    plt.xlim(right=true_exc_pulse['Time'][0])
    plt.xlabel('Time')
    plt.ylabel('Excitation Pulse')
    plt.title('True Excitation Pulse')

    plt.subplot(212)
    plt.axhline(y=predicted_data['B_x'][0], color='b', ls='-', linewidth=2, label='B_x')
    plt.axhline(y=predicted_data['B_y'][0], color='r', ls='-', linewidth=2, label='B_y')
    plt.xlim(right=true_exc_pulse['Time'][0])
    plt.xlabel('Time')
    plt.ylabel('Excitation Pulse')
    plt.title('Predicted Excitation Pulse')

    plt.tight_layout()
    plt.show()
'''base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/2const_exc_data.csv").resolve()
true_exc_data = pd.read_csv(file_path)
plot_const_evol(true_exc_data)'''