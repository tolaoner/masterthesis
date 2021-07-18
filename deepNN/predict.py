import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pathlib as pb
from matplotlib import pyplot as plt

base_path = pb.Path(__file__).parent.parent.parent
file_path = (base_path / "datasets" / "3const_exc_test.csv").resolve()

test_data = pd.read_csv(file_path)
test_labels = test_data[['B_x', 'B_y']].copy()
test_features = test_data.drop(['B_x', 'B_y'], axis=1)

model = keras.models.load_model('models/600k_matlab_set')
model.summary()
loss, acc = model.evaluate(test_features, test_labels)