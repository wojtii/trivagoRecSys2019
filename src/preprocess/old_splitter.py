from pathlib import Path
import dask.dataframe as dd
import numpy as np

current_directory = Path(__file__).absolute().parent
data_directory = current_directory.joinpath("..", "..", "data")
train_src = data_directory.joinpath("train_idx.csv")
train_dst = data_directory.joinpath("train.csv")
test_dst = data_directory.joinpath("test.csv")
gt_dst = data_directory.joinpath("ground_truth.csv")
partition_val = 0.8

df = dd.read_csv(train_src)

sessions_ids = df.session_id.unique().values.compute()
session_partition_id = sessions_ids[int(partition_val * len(sessions_ids))]

partition_idx = df[df.session_id == session_partition_id].global_idx.values.compute()[0]
print(partition_idx)

df_train = df.loc[df.global_idx < partition_idx].drop("global_idx", axis=1)
df_test = df.loc[df.global_idx >= partition_idx].drop("global_idx", axis=1)

df_train.to_csv(train_dst, single_file=True)
df_test.to_csv(gt_dst, single_file=True)

df_test_pd = df_test.compute()
test_first_steps_idxs = df_test_pd[df_test_pd.step == 1].index
test_last_steps_idxs = [x - 1 for x in test_first_steps_idxs[1:]]
df_test_pd.loc[
    (df_test_pd.index.isin(test_last_steps_idxs))
    & (df_test_pd.action_type == "clickout item"),
    "reference",
] = np.nan

df_test_pd.to_csv(test_dst)
