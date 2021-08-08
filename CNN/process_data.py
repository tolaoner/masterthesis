import pandas as pd
from matplotlib import pyplot as plt
import pathlib as pb
pd.options.display.max_rows = 10
print('imported modules')
base_path = pb.Path(__file__).parent.parent.parent
file_path = (base_path / "datasets" / "3const_exc_data.csv").resolve()
data = pd.read_csv(file_path)
label_data = data[['B_x', 'B_y']].copy()
cnn_data = data.drop(['Time', 'B_z', 'B_x', 'B_y'], axis=1)
# cnn_data['voxel_1'] = cnn_data['Mx_IC_1'].astype(str) + cnn_data['Mx_IC_1'].astype(str)
cnn_data = cnn_data.to_numpy()

print(cnn_data)