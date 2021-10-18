import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pathlib as pb
pd.options.display.max_rows = 10
print('imported modules')
cnn_data = np.load('cnn_array.npy')
#cnn_data = cnn_data.reshape(20000,10, 10, 7)
print(cnn_data.shape)