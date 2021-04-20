import pandas as pd
import numpy as np
import matplotlib as plt
import pathlib as pt
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
base_path = pt.Path(__file__).parent.parent.parent
data_path = (base_path / "data_generation/const_exc_data.csv").resolve()
df_data = pd.read_csv(data_path)
df_X = df_data.drop(['B_x'], axis='columns')
train_data = df_data.sample(frac=0.9, random_state=200)
test_data = df_data.drop(train_data.index)
# reset the indexes
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
# separate label and features for train data
train_label = train_data['B_x']
train_features = train_data.drop(['B_x'], axis='columns')
# separate features and labels in test data
test_label = test_data['B_x']
test_features = test_data.drop(['B_x'], axis='columns')
# create model
embedder = RandomTreesEmbedding(n_estimators=800,
                                max_depth=7,
                                min_samples_split=10,
                                n_jobs=1,
                                random_state=42).fit(df_X)
train_features = embedder.transform(train_features)
test_features = embedder.transform(test_features)
model = Ridge(alpha=3000).fit(train_features, train_label)
predictions = model.predict(test_features)
mean_squared_error = np.mean(abs(predictions**2-test_label**2))
print(mean_squared_error)

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