import pathlib as pt
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak

base_path = pt.Path(__file__).parent.parent.parent
data_path = (base_path / "data_generation/const_exc_data.csv").resolve()
df_data = pd.read_csv(data_path)
train_data = df_data.sample(frac=0.9, random_state=200)
test_data = df_data.drop(train_data.index)
# reset the indexes
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
x_train = train_data
y_train = x_train.pop("B_x")
x_test = test_data
y_test = x_test.pop("B_x")

reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=3
)
reg.fit(
    x_train,
    y_train,
    epochs=7
)
predictions = reg.predict(x_test)
print(reg.evaluate(x_test, y_test))
model = reg.export_model()
model.summary()
