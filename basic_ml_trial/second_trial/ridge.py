import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import Ridge
import pathlib as pt
from sklearn.model_selection import train_test_split
base_path = pt.Path(__file__).parent.parent.parent
data_path = (base_path / "data_generation/const_exc_data.csv").resolve()
df_data = pd.read_csv(data_path)
train = df_data.sample(frac=0.9, random_state=200)
test = df_data.drop(train.index)
# reset the indexes
train_data = train.reset_index()
test_data = test.reset_index()
# separate label and features for train data
train_label = train_data['B_x']
train_features = train_data.drop(['B_x'], axis='columns')
# create model
model = Ridge(alpha=3000).fit(train_features, train_label)
# separate features and labels in test data
test_label = test_data['B_x']
test_features = test_data.drop(['B_x'], axis='columns')
# test model on test data
predictions = model.predict(test_features)
# calculate error
mean_squared_error = np.mean(abs(predictions**2-test_label**2))
print(mean_squared_error)