from pathlib import Path
import pandas as pd

current_directory = Path(__file__).absolute().parent
data_directory = current_directory.joinpath("..", "..", "data")
train_src = data_directory.joinpath("train_src.csv")
train_dst = data_directory.joinpath("train_idx.csv")

df = pd.read_csv(train_src)
df["global_idx"] = df.index
df.to_csv(train_dst)
