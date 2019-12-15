from pathlib import Path
import pandas as pd
import numpy as np
import time

current_directory = Path(__file__).absolute().parent
data_directory = current_directory.joinpath("..", "..", "data")
train_src = data_directory.joinpath("train_src.csv")
train_dst = data_directory.joinpath("train.csv")
test_dst = data_directory.joinpath("test.csv")
gt_dst = data_directory.joinpath("ground_truth.csv")
partition_val = 0.8

t = time.time()
df = pd.read_csv(train_src)

sessions_ids = df.session_id.unique()
session_partition_id = sessions_ids[int(partition_val * len(sessions_ids))]
partition_idx = df[(df.session_id == session_partition_id)].index[0]
print(f"t1: {time.time()-t}")

t = time.time()
df_train = df[:partition_idx]
df_test = df[partition_idx:]
print(df_train.shape[0] / (df_train.shape[0] + df_test.shape[0]) * 100)
df_train.to_csv(train_dst)
df_test.to_csv(gt_dst)
print(f"t2: {time.time()-t}")

t = time.time()
test_first_steps_idxs = df_test[df.step == 1].index
test_last_steps_idxs = [x - 1 for x in test_first_steps_idxs[1:]]
df_test.loc[
    (df_test.index.isin(test_last_steps_idxs))
    & (df_test.action_type == "clickout item"),
    "reference",
] = np.nan

df_test.to_csv(test_dst)

print(f"t4: {time.time()-t}")
