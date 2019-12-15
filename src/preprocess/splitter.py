from pathlib import Path

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

current_directory = Path(__file__).absolute().parent
data_directory = current_directory.joinpath("..", "..", "data")
train_src = data_directory.joinpath("train_src.csv")
train_dst = data_directory.joinpath("train_parquet")
test_dst = data_directory.joinpath("test_parquet")
gt_dst = data_directory.joinpath("ground_truth_parquet")
split_val = 0.8

pbar = ProgressBar()
pbar.register()

df = dd.read_csv(train_src).repartition(npartitions=10)

part_split_idx = int(df.npartitions * split_val)
df_part_split = df.get_partition(part_split_idx)
part_row_idx = df_part_split[df_part_split.step == 1].compute().index[0]

df_part_split_train = df_part_split.loc[df_part_split.index < part_row_idx]
df_part_split_test = df_part_split.loc[df_part_split.index >= part_row_idx]

df_train = df.partitions[:part_split_idx].append(df_part_split_train)
# df_train.to_csv(train_dst, single_file=True)
# dd.to_parquet(df_train.repartition(npartitions=1), train_dst)
df_test_pd = (
    df.partitions[part_split_idx + 1 :]
    .append(df_part_split_test)
    .compute()
    .reset_index(drop=True)
)
# df_test_pd.to_csv(gt_dst)
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_pandas(df_test_pd)
pq.write_table(table, str(gt_dst))

df_test_pd.to_parquet(gt_dst)
exit(0)

test_first_steps_idxs = (
    df_test_pd[df_test_pd.step == 1].drop_duplicates("session_id").index
)
test_last_steps_idxs = {i - 1 for i in test_first_steps_idxs}

df_test_pd.loc[
    (df_test_pd.action_type == "clickout item")
    & (df_test_pd.index.isin(test_last_steps_idxs))
    & ~(df_test_pd.duplicated(subset="session_id", keep="last")),
    "reference",
] = ""

# df_test_pd.to_csv(test_dst)
df_test_pd.to_parquet(test_dst)

pbar.unregister()
