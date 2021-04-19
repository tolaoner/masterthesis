import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn
import pathlib as pt
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
base_path = pt.Path(__file__).parent.parent.parent
data_path = (base_path / "data_generation/const_exc_data.csv").resolve()
df_data = pd.read_csv(data_path)
train = df_data.sample(frac=0.9, random_state=200)
test = df_data.drop(train.index)
train_data = train.reset_index()
test_data = test.reset_index()
'''
X_train = train_data.iloc[:, 1:-1].to_numpy()
X_test = test_data.iloc[:, 1:].to_numpy()
X = np.vstack([X_train, X_test])
y = train_data['target'].to_numpy()

embedder = RandomTreesEmbedding(
    n_estimators=800,
    max_depth=7,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
).fit(X)

X_train = embedder.transform(X_train)
X_test = embedder.transform(X_test)
model = Ridge(alpha=3000).fit(X_train, y)

sub = pd.read_csv('../input/tabular-playground-series-jan-2021/sample_submission.csv')
sub['target'] = model.predict(X_test)
sub.to_csv('submission.csv', index=False)
'''