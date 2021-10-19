import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pathlib as pb
import plot_const_pulse as pcp
from matplotlib import pyplot as plt

base_path = pb.Path(__file__).parent.parent.parent
file_path = (base_path / "datasets" / "const_2vox" / "norm_2const_exc.csv").resolve()
file_path3 = (base_path / "datasets" / "const_2vox" / "2const_exc_data.csv").resolve()

test_data = pd.read_csv(file_path)
feature_data = pd.read_csv(file_path3)

test_labels = feature_data[['Time', 'B_z', 'B_x', 'B_y']].copy()
test_features = test_data.drop(['B_x', 'B_y'], axis=1)
test_features_np = np.array(test_features)
#print(test_labels)
#print(test_features)
model_path = (base_path / "masterthesis" / "deepNN" / "models" / "600k_norm_matlab_set").resolve()
model = keras.models.load_model(model_path)
#model.summary()
# loss, acc = model.evaluate(test_features[0], test_labels[0])

predictions = model.predict(test_features_np[0:1])
predictions_pd = pd.DataFrame(predictions, columns=['B_x', 'B_y'])


file_path2 = (base_path / "datasets" / "const_2vox" / "norm_2const_exc.csv").resolve()
evaluation_data = pd.read_csv(file_path2)
evaluation_data = evaluation_data.sample(n=6000)
#print(evaluation_data)
y_test = evaluation_data[['B_x', 'B_y']].copy()
x_test = evaluation_data.drop(['B_x', 'B_y'], axis=1)
results = model.evaluate(x_test, y_test, batch_size=40)
print("Test loss, test rmse:", results)

pcp.plot_const_evol(test_labels, predictions_pd)