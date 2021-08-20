import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pathlib as pb
import plot_const_pulse as pcp
from matplotlib import pyplot as plt

base_path = pb.Path(__file__).parent.parent.parent
file_path = (base_path / "datasets" / "2const_exc_data.csv").resolve()

test_data = pd.read_csv(file_path)

test_labels = test_data[['Time', 'B_z', 'B_x', 'B_y']].copy()
test_features = test_data.drop(['B_x', 'B_y'], axis=1)
test_features_np = np.array(test_features)

model = keras.models.load_model('models/trained_matlab_set')
# model.summary()
# loss, acc = model.evaluate(test_features[0], test_labels[0])

predictions = model.predict(test_features_np[0:1])
predictions_pd = pd.DataFrame(predictions, columns=['B_x', 'B_y'])
pcp.plot_const_evol(test_labels, predictions_pd)