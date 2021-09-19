import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/dc_timevarying.csv").resolve()
data = pd.read_csv(file_path)
t_span = np.linspace(0, 0.001, 21)
mag_f = data[['Mx_f', 'My_f', 'Mz_f']].copy()
plt.figure()
plt.plot(t_span, mag_f['Mx_f'], label="Progress of Magnetization")
plt.plot(t_span, mag_f['My_f'])
plt.plot(t_span, mag_f['Mz_f'])
plt.xlabel('time')
plt.ylabel('Magnetization')
plt.legend()
plt.show()