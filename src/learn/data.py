import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from ordered_set import OrderedSet
from sklearn.preprocessing import MinMaxScaler, StandardScaler

current_directory = Path(__file__).absolute().parent
data_directory = current_directory.joinpath("..", "..", "data")
train_src = data_directory.joinpath("train.csv")
test_src = data_directory.joinpath("test.csv")
NROWS = 1000000
DUMMY_ITEM = -1
DUMMY_ACTION = None
DUMMY_IMPRESSION_INDEX = 0


def compute_rank(inp):
    return [sorted(inp).index(i) for i in inp]


class CategoricalEncoder:
    def __init__(self):
        self.n_elements = 0
        self.f_dict = {}
        self.r_dict = {}

    def fit(self, array):
        unique_elements = OrderedSet(array)
        self.n_elements = 0
        self.f_dict = {}
        self.r_dict = {}

        for e in unique_elements:
            self.f_dict[e] = self.n_elements
            self.r_dict[self.n_elements] = e
            self.n_elements += 1

    def transform(self, array, to_np=False):
        transformed_array = [self.f_dict[e] for e in array]
        if to_np:
            transformed_array = np.array(transformed_array)
        return transformed_array

    def fit_transform(self, array, to_np=False):
        self.fit(array)
        return self.transform(array, to_np)


class NNDataLoader:
    def __init__(self, data, config, shuffle=True, batch_size=128, continuous_features=None):
        self.item_id = torch.LongTensor(data.item_id.values)
        self.config = config
        self.label = torch.FloatTensor(data.label.values)
        self.past_interactions = torch.LongTensor(np.vstack(data.past_interactions.values))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.item_id))
        self.past_interaction_masks = (self.past_interactions != self.config.transformed_dummy_item)
        self.price_rank = torch.LongTensor(data.price_rank.values)
        self.city = torch.LongTensor(data.city.values)
        self.last_item = torch.LongTensor(data.last_item.values)
        self.impression_index = torch.LongTensor(data.impression_index)
        self.continuous_features = torch.FloatTensor(
            data.loc[:, continuous_features].values
        )
        self.neighbor_prices = torch.FloatTensor(np.vstack(data.neighbor_prices))
        self.past_interactions_sess = torch.LongTensor(
            np.vstack(data.past_interactions_sess.values)
        )
        self.past_actions_sess = torch.LongTensor(
            np.vstack(data.past_actions_sess.values)
        )
        self.last_click_item = torch.LongTensor(data.last_click_item.values)
        self.last_click_impression = torch.LongTensor(data.last_click_impression.values)
        self.last_interact_index = torch.LongTensor(data.last_interact_index.values)
        self.city_platform = torch.LongTensor(data.city_platform.values)

    def __len__(self):
        return len(self.item_id) // self.batch_size

    def __iter__(self):
        self.batch_id = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.batch_id * self.batch_size > len(self.indices):
            raise StopIteration

        current_indices = self.indices[self.batch_id * self.batch_size: (self.batch_id + 1) * self.batch_size]
        self.batch_id += 1
        return [
            self.item_id[current_indices],
            self.label[current_indices],
            self.past_interactions[current_indices],
            self.past_interaction_masks[current_indices],
            self.price_rank[current_indices],
            self.city[current_indices],
            self.last_item[current_indices],
            self.impression_index[current_indices],
            self.continuous_features[current_indices],
            self.past_interactions_sess[current_indices],
            self.past_actions_sess[current_indices],
            self.last_click_item[current_indices],
            self.last_click_impression[current_indices],
            self.last_interact_index[current_indices],
            self.neighbor_prices[current_indices],
            self.city_platform[current_indices],
        ]


class NNDataGenerator:
    def __init__(self, config):
        self.config = config
        self.target_action = self.config.target_action = "clickout item"
        self.config.all_cat_columns = self.all_cat_columns = [
            "user_id",
            "item_id",
            "city",
            "action",
            "city_platform",
        ]

        train = pd.read_csv(train_src, nrows=NROWS)
        train["id"] = np.arange(len(train))

        test = pd.read_csv(test_src, nrows=NROWS)
        test["id"] = np.arange(len(train), len(train) + len(test))

        train.rename(columns={"reference": "item_id", "action_type": "action"}, inplace=True)
        test.rename(columns={"reference": "item_id", "action_type": "action"}, inplace=True)
        train["in_impressions"] = True
        train.loc[~train.impressions.isna(), "in_impressions"] = train.loc[
            ~train.impressions.isna()
        ].apply(lambda row: row.item_id in row.impressions.split("|"), axis=1)
        train = train.loc[train.in_impressions].drop("in_impressions", axis=1).reset_index(drop=True)
        test["in_impressions"] = True
        test.loc[(~test.impressions.isna()) & (~test.item_id.isna()), "in_impressions"] = test.loc[
            (~test.impressions.isna()) & (~test.item_id.isna())].apply(
            lambda row: row.item_id in row.impressions.split("|"), axis=1)
        test = test.loc[test.in_impressions].drop("in_impressions", axis=1).reset_index(drop=True)

        train["item_id"] = train["item_id"].apply(str)
        train.loc[~train.impressions.isna(), "impressions"] = train.loc[~train.impressions.isna()].impressions.apply(
            lambda x: x.split("|"))
        train.loc[~train.prices.isna(), "prices"] = train.loc[~train.prices.isna()].prices.apply(
            lambda x: x.split("|")).apply(lambda x: [int(p) for p in x])

        test["item_id"] = test["item_id"].apply(str)
        test.loc[~test.impressions.isna(), "impressions"] = test.loc[~test.impressions.isna()].impressions.apply(
            lambda x: x.split("|"))
        test.loc[~test.prices.isna(), "prices"] = test.loc[~test.prices.isna()].prices.apply(
            lambda x: x.split("|")).apply(lambda x: [int(p) for p in x])

        data = pd.concat([train, test], axis=0)
        data = data.reset_index(drop=True)
        all_items = []

        for imp in data.loc[~data.impressions.isna()].impressions.tolist() + [data.item_id.apply(str).tolist()]:
            all_items += imp

        unique_items = OrderedSet(all_items)
        unique_actions = OrderedSet(data.action.values)

        train_session_interactions = dict(train.groupby("session_id")["item_id"].apply(list))
        test_session_interactions = dict(test.groupby("session_id")["item_id"].apply(list))
        train_session_actions = dict(train.groupby("session_id")["action"].apply(list))
        test_session_actions = dict(test.groupby("session_id")["action"].apply(list))

        train["sess_step"] = train.groupby("session_id")["timestamp"].rank(method="max").apply(int)
        test["sess_step"] = test.groupby("session_id")["timestamp"].rank(method="max").apply(int)

        train["city_platform"] = train.apply(lambda x: x["city"] + x["platform"], axis=1)
        test["city_platform"] = test.apply(lambda x: x["city"] + x["platform"], axis=1)
        train["last_item"] = np.nan
        test["last_item"] = np.nan

        train_shifted_item_id = [DUMMY_ITEM] + train.item_id.values[:-1].tolist()
        test_shifted_item_id = [DUMMY_ITEM] + test.item_id.values[:-1].tolist()
        train["last_item"] = train_shifted_item_id
        test["last_item"] = test_shifted_item_id

        train["step_rank"] = train.groupby("session_id")["timestamp"].rank(method="max", ascending=True)
        test["step_rank"] = test.groupby("session_id")["timestamp"].rank(method="max", ascending=True)

        train.loc[(train.step_rank == 1), "last_item"] = DUMMY_ITEM
        test.loc[(test.step_rank == 1), "last_item"] = DUMMY_ITEM

        data = pd.concat([train, test], axis=0)
        data = data.reset_index(drop=True)

        data_feature = data.loc[:, ["id", "session_id", "timestamp", "step"]].copy()
        data_feature = data_feature.drop(["session_id", "timestamp", "step"], axis=1)
        train = train.merge(data_feature, on="id", how="left")
        test = test.merge(data_feature, on="id", how="left")

        self.cat_encoders = {col: CategoricalEncoder() for col in self.all_cat_columns}
        self.cat_encoders["item_id"].fit(list(unique_items) + [DUMMY_ITEM])
        self.cat_encoders["city"].fit(data.city.values)
        self.cat_encoders["city_platform"].fit(data.city_platform.values)
        self.cat_encoders["action"].fit(list(unique_actions) + [DUMMY_ACTION])
        self.cat_encoders["user_id"].fit(data.user_id.values)

        for col in self.all_cat_columns:
            train[col] = self.cat_encoders[col].transform(train[col].values)
            test[col] = self.cat_encoders[col].transform(test[col].values)
            self.config.num_embeddings[col] = self.cat_encoders[col].n_elements

        self.config.transformed_clickout_action = (self.transformed_clickout_action
                                                   ) = self.cat_encoders["action"].transform(["clickout item"])[0]
        self.config.transformed_dummy_action = (
            self.transformed_dummy_action
        ) = self.cat_encoders["action"].transform([DUMMY_ACTION])[0]
        self.transformed_interaction_deals = self.cat_encoders["action"].transform(
            ["interaction item deals"]
        )[0]
        self.transformed_interaction_info = self.cat_encoders["action"].transform(
            ["interaction item info"]
        )[0]
        self.transformed_interaction_rating = self.cat_encoders["action"].transform(
            ["interaction item rating"]
        )[0]

        self.config.transformed_dummy_item = (
            self.transformed_dummy_item
        ) = self.cat_encoders["item_id"].transform([DUMMY_ITEM])[0]
        self.config.transformed_nan_item = (
            self.transformed_nan_item
        ) = self.cat_encoders["item_id"].transform(["nan"])[0]

        train["last_item"] = self.cat_encoders["item_id"].transform(
            train["last_item"].values
        )
        test["last_item"] = self.cat_encoders["item_id"].transform(
            test["last_item"].values
        )

        for session_id, item_list in train_session_interactions.items():
            train_session_interactions[session_id] = [
                                                         self.transformed_dummy_item
                                                     ] * self.config.sess_length + self.cat_encoders[
                                                         "item_id"].transform(
                item_list
            )

        for session_id, item_list in test_session_interactions.items():
            test_session_interactions[session_id] = [
                                                        self.transformed_dummy_item
                                                    ] * self.config.sess_length + self.cat_encoders[
                                                        "item_id"].transform(
                item_list
            )

        for session_id, action_list in train_session_actions.items():
            train_session_actions[session_id] = [
                                                    self.transformed_dummy_action
                                                ] * self.config.sess_length + self.cat_encoders["action"].transform(
                action_list
            )

        for session_id, action_list in test_session_actions.items():
            test_session_actions[session_id] = [
                                                   self.transformed_dummy_action
                                               ] * self.config.sess_length + self.cat_encoders["action"].transform(
                action_list
            )

        implicit_train = train.loc[train.action != self.transformed_clickout_action, :]
        implicit_test = test.loc[test.action != self.transformed_clickout_action, :]

        interaction_item_ids = (
                implicit_train.drop_duplicates(
                    subset=["session_id", "item_id", "action"]
                ).item_id.tolist()
                + implicit_test.drop_duplicates(
            subset=["session_id", "item_id", "action"]
        ).item_id.tolist()
        )
        unique_interaction_items, counts = np.unique(
            interaction_item_ids, return_counts=True
        )
        self.interaction_count_dict = dict(zip(unique_interaction_items, counts))

        train = train.loc[train.action == self.transformed_clickout_action, :]
        test = test.loc[test.action == self.transformed_clickout_action, :]

        train["step_rank"] = train.groupby("session_id")["step"].rank(
            method="max", ascending=False
        )

        item_ids = np.hstack(
            [np.hstack(train["impressions"].values), np.hstack(test.impressions.values)]
        )

        unique_items, counts = np.unique(item_ids, return_counts=True)
        self.item_popularity_dict = dict(zip(unique_items, counts))

        clickout_item_ids = (
                train.drop_duplicates(
                    subset=["session_id", "item_id", "action"]
                ).item_id.tolist()
                + test.drop_duplicates(
            subset=["session_id", "item_id", "action"]
        ).item_id.tolist()
        )
        unique_clickout_items, counts = np.unique(clickout_item_ids, return_counts=True)

        self.clickout_count_dict = dict(zip(unique_clickout_items, counts))
        val = train.loc[train.step_rank == 1, :].iloc[:50000]
        train = train.loc[~train.index.isin(train.loc[train.step_rank == 1, :].iloc[:50000].index), :]

        self.past_interaction_dict = {}
        self.last_click_sess_dict = {}
        self.last_impressions_dict = {}
        self.sess_last_imp_idx_dict = {}
        self.sess_last_price_dict = {}

        self.train_data = self.build_user_item_interactions(
            train,
            train_session_interactions,
            train_session_actions,
        )
        self.val_data = self.build_user_item_interactions(
            val,
            train_session_interactions,
            train_session_actions,
        )
        self.test_data, labeled_test = self.build_user_item_interactions(
            test,
            test_session_interactions,
            test_session_actions,
            training=False,
        )

        price_sc = StandardScaler()
        self.train_data["price_diff"] = price_sc.fit_transform(
            self.train_data.price_diff.values.reshape(-1, 1)
        )
        self.val_data["price_diff"] = price_sc.transform(
            self.val_data.price_diff.values.reshape(-1, 1)
        )
        self.test_data["price_diff"] = price_sc.transform(
            self.test_data.price_diff.values.reshape(-1, 1)
        )

        price_mm = MinMaxScaler()
        self.train_data["price_ratio"] = price_mm.fit_transform(
            self.train_data.price_ratio.values.reshape(-1, 1)
        )
        self.val_data["price_ratio"] = price_mm.transform(
            self.val_data.price_ratio.values.reshape(-1, 1)
        )
        self.test_data["price_ratio"] = price_mm.transform(
            self.test_data.price_ratio.values.reshape(-1, 1)
        )
        price_mm.fit(
            np.hstack(
                [
                    np.hstack(self.train_data.neighbor_prices.values),
                    np.hstack(self.val_data.neighbor_prices.values),
                    np.hstack(self.test_data.neighbor_prices.values),
                ]
            ).reshape(-1, 1)
        )
        self.train_data["neighbor_prices"] = self.train_data["neighbor_prices"].apply(
            lambda x: price_mm.transform(np.array(x).reshape(-1, 1)).reshape(-1)
        )
        self.val_data["neighbor_prices"] = self.val_data["neighbor_prices"].apply(
            lambda x: price_mm.transform(np.array(x).reshape(-1, 1)).reshape(-1)
        )
        self.test_data["neighbor_prices"] = self.test_data["neighbor_prices"].apply(
            lambda x: price_mm.transform(np.array(x).reshape(-1, 1)).reshape(-1)
        )

        filter_df = data.loc[~data.current_filters.isna(), ["id", "current_filters"]]
        filter_df["current_filters"] = filter_df.current_filters.apply(
            lambda x: x.split("|")
        )
        filter_set = list(OrderedSet(np.hstack(filter_df["current_filters"].tolist())))

        self.cat_encoders["filters"] = CategoricalEncoder()
        self.cat_encoders["filters"].fit(filter_set)
        self.continuous_features = [
            "price_diff",
            "price",
            "price_ratio",
            "global_clickout_count_rank",
        ]

        self.train_data["global_clickout_count_rank"] /= 25
        self.val_data["global_clickout_count_rank"] /= 25
        self.test_data["global_clickout_count_rank"] /= 25

        for up in self.continuous_features:
            mean_value = self.train_data.loc[~self.train_data[up].isna(), up].mean()
            self.train_data.loc[:, up].fillna(mean_value, inplace=True)
            self.val_data.loc[:, up].fillna(mean_value, inplace=True)
            self.test_data.loc[:, up].fillna(mean_value, inplace=True)

        self.config.num_embeddings["price_rank"] = 25
        self.config.num_embeddings["impression_index"] = 26
        self.config.all_cat_columns += ["price_rank", "impression_index"]
        self.config.continuous_size = len(self.continuous_features)
        self.config.neighbor_size = 5
        self.all_cat_columns = self.config.all_cat_columns

        print(f"Shape of training data: {self.train_data.shape}")
        print(f"Shape of validation data: {self.val_data.shape}")
        print(f"Shape of test data: {self.test_data.shape}")

    def build_user_item_interactions(
            self,
            df,
            session_interactions,
            session_actions,
            training=True,
    ):
        df_list = []
        label_test_df_list = []
        for idx, row in enumerate(tqdm(df.itertuples())):
            if row.user_id not in self.past_interaction_dict:
                self.past_interaction_dict[row.user_id] = [self.transformed_dummy_item] * self.config.sequence_length
            if row.session_id not in self.last_click_sess_dict:
                self.last_click_sess_dict[row.session_id] = self.transformed_dummy_item
            if row.session_id not in self.last_impressions_dict:
                self.last_impressions_dict[row.session_id] = None
            if row.session_id not in self.sess_last_imp_idx_dict:
                self.sess_last_imp_idx_dict[row.session_id] = DUMMY_IMPRESSION_INDEX
            if row.session_id not in self.sess_last_price_dict:
                self.sess_last_price_dict[row.session_id] = None

            transformed_impressions = self.cat_encoders["item_id"].transform(
                row.impressions, to_np=True
            )
            sess_step = row.sess_step
            session_id = row.session_id

            current_session_interactions = session_interactions[session_id][
                                           : self.config.sess_length + sess_step - 1
                                           ]
            current_session_interactions = current_session_interactions[
                                           -self.config.sess_length:
                                           ]

            current_session_actions = session_actions[session_id][
                                      : self.config.sess_length + sess_step - 1
                                      ]
            current_session_actions = current_session_actions[
                                      -self.config.sess_length:
                                      ]

            if row.last_item in transformed_impressions:
                last_interact_index = transformed_impressions.tolist().index(
                    row.last_item
                )
            else:
                last_interact_index = DUMMY_IMPRESSION_INDEX

            label = transformed_impressions == row.item_id
            interaction_deals_indices = (
                    np.array(session_actions[session_id][
                             : self.config.sess_length + sess_step - 1]) == self.transformed_interaction_deals
            )
            interaction_deals_item = \
                np.array(session_interactions[session_id][: self.config.sess_length + sess_step - 1])[
                    interaction_deals_indices]
            unleaked_clickout_count = [
                self.clickout_count_dict[imp] if imp in self.clickout_count_dict else 0
                for imp in transformed_impressions
            ]
            unleaked_clickout_count = [
                unleaked_clickout_count[idx] - 1
                if imp == row.item_id
                else unleaked_clickout_count[idx]
                for idx, imp in enumerate(transformed_impressions)
            ]

            other_is_interacted = [imp in session_interactions[session_id][: self.config.sess_length + sess_step - 1]
                                   for imp in transformed_impressions]
            padded_other_is_interacted = other_is_interacted + [False] * (25 - len(other_is_interacted))

            other_is_clicked = [imp in self.past_interaction_dict[row.user_id] for imp in transformed_impressions]
            padded_other_is_clicked = other_is_clicked + [False] * (25 - len(other_is_clicked))

            padded_prices = [row.prices[0]] * 2 + row.prices + [row.prices[-1]] * 2
            price_rank = compute_rank(row.prices)
            current_rows = np.zeros([len(row.impressions), 24], dtype=object)
            current_rows[:, 0] = row.user_id
            current_rows[:, 1] = transformed_impressions
            current_rows[:, 2] = label
            current_rows[:, 3] = row.session_id
            current_rows[:, 4] = [
                                     np.array(self.past_interaction_dict[row.user_id])
                                 ] * len(row.impressions)
            current_rows[:, 5] = price_rank
            current_rows[:, 6] = row.city
            current_rows[:, 7] = row.last_item
            current_rows[:, 8] = np.arange(len(transformed_impressions))
            current_rows[:, 9] = row.step
            current_rows[:, 10] = row.id
            current_rows[:, 11] = [np.array(current_session_interactions)] * len(
                row.impressions
            )
            current_rows[:, 12] = [np.array(current_session_actions)] * len(
                row.impressions
            )
            current_rows[:, 13] = row.prices
            current_rows[:, 14] = self.last_click_sess_dict[row.session_id]
            current_rows[:, 15] = self.sess_last_imp_idx_dict[row.session_id]
            current_rows[:, 16] = last_interact_index
            current_rows[:, 17] = (
                row.prices - self.sess_last_price_dict[row.session_id]
                if self.sess_last_price_dict[row.session_id]
                else 0
            )

            current_rows[:, 18] = (
                row.prices / self.sess_last_price_dict[row.session_id]
                if self.sess_last_price_dict[row.session_id]
                else 0
            )
            current_rows[:, 19] = [
                padded_prices[i: i + 5] for i in range(len(row.impressions))
            ]

            current_rows[:, 20] = row.city_platform
            current_rows[:, 21] = [np.array(padded_other_is_interacted)] * len(
                row.impressions
            )
            current_rows[:, 22] = [np.array(padded_other_is_clicked)] * len(
                row.impressions
            )
            current_rows[:, 23] = np.argsort(np.argsort(unleaked_clickout_count))

            if training or row.item_id == self.transformed_nan_item:
                df_list.append(current_rows)
            else:
                label_test_df_list.append(current_rows)
            self.past_interaction_dict[row.user_id] = self.past_interaction_dict[
                                                          row.user_id
                                                      ][1:]
            self.past_interaction_dict[row.user_id].append(row.item_id)

            self.last_click_sess_dict[row.session_id] = row.item_id
            self.last_impressions_dict[
                row.session_id
            ] = transformed_impressions.tolist()

            if row.item_id != self.transformed_nan_item:
                self.sess_last_imp_idx_dict[row.session_id] = (
                    (transformed_impressions == row.item_id).tolist().index(True)
                )
                self.sess_last_price_dict[row.session_id] = np.array(row.prices)[
                    transformed_impressions == row.item_id
                    ][0]

        data = np.vstack(df_list)
        dtype_dict = {
            "city": "int32",
            "last_item": "int32",
            "impression_index": "int32",
            "step": "int32",
            "id": "int32",
            "user_id": "int32",
            "item_id": "int32",
            "label": "int32",
            "price_rank": "int32",
            "price": "float32",
            "last_click_item": "int32",
            "last_click_impression": "int16",
            "last_interact_index": "int16",
            "price_diff": "float32",
            "price_ratio": "float32",
            "city_platform": "int32",
            "global_clickout_count_rank": "int8",
        }
        df_columns = [
            "user_id",
            "item_id",
            "label",
            "session_id",
            "past_interactions",
            "price_rank",
            "city",
            "last_item",
            "impression_index",
            "step",
            "id",
            "past_interactions_sess",
            "past_actions_sess",
            "price",
            "last_click_item",
            "last_click_impression",
            "last_interact_index",
            "price_diff",
            "price_ratio",
            "neighbor_prices",
            "city_platform",
            "other_is_interacted",
            "other_is_clicked",
            "global_clickout_count_rank",
        ]
        df = pd.DataFrame(data, columns=df_columns)
        df = df.astype(dtype=dtype_dict)
        if training:
            return df
        else:
            label_test = np.vstack(label_test_df_list)
            label_test = pd.DataFrame(label_test, columns=df_columns)
            label_test = label_test.astype(dtype=dtype_dict)
            return df, label_test

    def instance_a_train_loader(self):
        train_data = self.train_data
        return NNDataLoader(
            train_data,
            self.config,
            shuffle=True,
            batch_size=self.config.batch_size,
            continuous_features=self.continuous_features,
        )

    def evaluate_data_valid(self):
        val_data = self.val_data
        return NNDataLoader(
            val_data,
            self.config,
            shuffle=False,
            batch_size=self.config.batch_size,
            continuous_features=self.continuous_features,
        )
