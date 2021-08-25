import pandas as pd
import numpy as np
from pathlib import Path

base_path = Path(__file__).parent.parent.parent
file_path = (base_path / "datasets/1const_exc_data.csv").resolve()

data = pd.read_csv(file_path)
label = data[['B_z', 'B_x', 'B_y']].copy()
time_data = data['Time'].copy()
features = data.drop(['Time', 'B_z', 'B_x', 'B_y'], axis=1)
time_data = time_data*1000

for column in features:
    features[column] = features[column]/features[column].max()

print(features)
dataset = pd.concat([time_data, features, label], axis=1)
print(dataset)
new_file_path = (base_path / "datasets/norm_1const_exc.csv").resolve()
dataset.to_csv(new_file_path, mode='w', index=False)