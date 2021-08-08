import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
import numpy as np
from matplotlib import pyplot as pl
from pathlib import Path
print('imported modules')
base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/cnn_array.npy").resolve()
train_data = np.load(file_path)
print(train_data)
