import pathlib as pt
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
h2o.init(
    nthreads=-1,     # number of threads when launching a new H2O server
    max_mem_size=12  # in gigabytes
)
# read the data from csv file
base_path = pt.Path(__file__).parent.parent.parent
data_path = (base_path / "data_generation/const_exc_data.csv").resolve()
train_as_df = pd.read_csv(data_path)
# convert dataframe to h20 frame
train_data = h2o.H2OFrame(train_as_df)
# split features from label
x = train_data.columns
y = "B_x"
x.remove(y)
# Run AutoMl for 10 base models
aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=x, y=y, training_frame=train_data)
# show Leaderboard (since we did not specify a leaderboard_frame in h2oautoml.train method,
# it uses cross-validation metrics to rank the models
lb = aml.leaderboard
lb.head(rows=lb.nrows)