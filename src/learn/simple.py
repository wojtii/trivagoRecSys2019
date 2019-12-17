# based on https://blog.rosetta.ai/the-5th-place-approach-to-the-2019-acm-recsys-challenge-by-team-rosettaai-eb3c4e6178c4
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ordered_set import OrderedSet
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tqdm import tqdm

current_directory = Path(__file__).absolute().parent
data_directory = current_directory.joinpath("..", "..", "data")
train_src = data_directory.joinpath("train.csv")
test_src = data_directory.joinpath("test.csv")
NROWS = 1000000
DUMMY_ITEM = -1
DUMMY_ACTION = None
DUMMY_IMPRESSION_INDEX = 0


class NNDataLoader:
    def __init__(self, data, config, shuffle=True, batch_size=128, continuous_features=None):
        self.reference = torch.LongTensor(data.reference.values)
        self.config = config
        self.label = torch.FloatTensor(data.label.values)
        self.past_interactions = torch.LongTensor(np.vstack(data.past_interactions.values))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.reference))
        self.past_interaction_masks = (self.past_interactions != self.config.transformed_dummy_item)
        self.price_rank = torch.LongTensor(data.price_rank.values)
        self.city = torch.LongTensor(data.city.values)
        self.last_item = torch.LongTensor(data.last_item.values)
        self.impression_index = torch.LongTensor(data.impression_index)
        self.continuous_features = torch.FloatTensor(data.loc[:, continuous_features].values)
        self.neighbor_prices = torch.FloatTensor(np.vstack(data.neighbor_prices))
        self.past_interactions_sess = torch.LongTensor(np.vstack(data.past_interactions_sess.values))
        self.past_actions_sess = torch.LongTensor(np.vstack(data.past_actions_sess.values))
        self.last_click_item = torch.LongTensor(data.last_click_item.values)
        self.last_interact_index = torch.LongTensor(data.last_interact_index.values)
        self.city_platform = torch.LongTensor(data.city_platform.values)

    def __len__(self):
        return len(self.reference) // self.batch_size

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
            self.reference[current_indices],
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
            self.last_interact_index[current_indices],
            self.neighbor_prices[current_indices],
            self.city_platform[current_indices],
        ]


class NNDataGenerator:
    def __init__(self, config):
        self.config = config
        train = pd.read_csv(train_src, nrows=NROWS)
        train["id"] = np.arange(len(train))
        test = pd.read_csv(test_src, nrows=NROWS)
        test["id"] = np.arange(len(train), len(train) + len(test))

        train.rename(columns={"reference": "reference", "action_type": "action"}, inplace=True)
        test.rename(columns={"reference": "reference", "action_type": "action"}, inplace=True)

        data = pd.concat([train, test], axis=0)
        data = data.reset_index(drop=True)
        all_items = []

        for imp in data.loc[~data.impressions.isna()].impressions.tolist() + [data.reference.apply(str).tolist()]:
            all_items += imp

        unique_items = OrderedSet(all_items)
        unique_actions = OrderedSet(data.action.values)

        train["sess_step"] = train.groupby("session_id")["timestamp"].rank(method="max").apply(int)
        test["sess_step"] = test.groupby("session_id")["timestamp"].rank(method="max").apply(int)

        train["city_platform"] = train.apply(lambda x: x["city"] + x["platform"], axis=1)
        test["city_platform"] = test.apply(lambda x: x["city"] + x["platform"], axis=1)
        train["last_item"] = np.nan
        test["last_item"] = np.nan

        train_shifted_item_id = [DUMMY_ITEM] + train.reference.values[:-1].tolist()
        test_shifted_item_id = [DUMMY_ITEM] + test.reference.values[:-1].tolist()
        train["last_item"] = train_shifted_item_id
        test["last_item"] = test_shifted_item_id

        train["step_rank"] = train.groupby("session_id")["timestamp"].rank(method="max", ascending=True)
        test["step_rank"] = test.groupby("session_id")["timestamp"].rank(method="max", ascending=True)

        data = pd.concat([train, test], axis=0)
        data = data.reset_index(drop=True)

        data_feature = data.loc[:, ["id", "session_id", "timestamp", "step"]].copy()
        data_feature = data_feature.drop(["session_id", "timestamp", "step"], axis=1)
        train = train.merge(data_feature, on="id", how="left")
        test = test.merge(data_feature, on="id", how="left")

        self.cat_encoders = {col: LabelEncoder() for col in self.all_cat_columns}
        self.cat_encoders["reference"].fit(list(unique_items) + [DUMMY_ITEM])
        self.cat_encoders["city"].fit(data.city.values)
        self.cat_encoders["city_platform"].fit(data.city_platform.values)
        self.cat_encoders["action"].fit(list(unique_actions) + [DUMMY_ACTION])
        self.cat_encoders["user_id"].fit(data.user_id.values)

        for col in self.all_cat_columns:
            train[col] = self.cat_encoders[col].transform(train[col].values)
            test[col] = self.cat_encoders[col].transform(test[col].values)
            self.config.num_embeddings[col] = self.cat_encoders[col].classes_.size

        self.config.transformed_clickout_action = (self.transformed_clickout_action
                                                   ) = self.cat_encoders["action"].transform(["clickout item"])[0]
        self.config.transformed_dummy_action = self.cat_encoders["action"].transform([DUMMY_ACTION])[0]
        self.transformed_interaction_deals = self.cat_encoders["action"].transform(["interaction item deals"])[0]
        self.transformed_interaction_info = self.cat_encoders["action"].transform(["interaction item info"])[0]
        self.transformed_interaction_rating = self.cat_encoders["action"].transform(["interaction item rating"])[0]

        self.config.transformed_dummy_item = (
            self.transformed_dummy_item
        ) = self.cat_encoders["reference"].transform([DUMMY_ITEM])[0]
        self.config.transformed_nan_item = (
            self.transformed_nan_item
        ) = self.cat_encoders["reference"].transform(["nan"])[0]

        train["last_item"] = self.cat_encoders["reference"].transform(train["last_item"].values)
        test["last_item"] = self.cat_encoders["reference"].transform(test["last_item"].values)

        implicit_train = train.loc[train.action != self.transformed_clickout_action, :]
        implicit_test = test.loc[test.action != self.transformed_clickout_action, :]

        interaction_item_ids = implicit_train.drop_duplicates(
            subset=["session_id", "reference", "action"]
        ).reference.tolist() + implicit_test.drop_duplicates(
            subset=["session_id", "reference", "action"]).reference.tolist()

        unique_interaction_items, counts = np.unique(
            interaction_item_ids, return_counts=True
        )
        self.interaction_count_dict = dict(zip(unique_interaction_items, counts))
        train = train.loc[train.action == self.transformed_clickout_action, :]
        test = test.loc[test.action == self.transformed_clickout_action, :]

        train["step_rank"] = train.groupby("session_id")["step"].rank(method="max", ascending=False)
        item_ids = np.hstack([np.hstack(train["impressions"].values), np.hstack(test.impressions.values)])
        unique_items, counts = np.unique(item_ids, return_counts=True)
        self.item_popularity_dict = dict(zip(unique_items, counts))
        clickout_item_ids = train.drop_duplicates(
            subset=["session_id", "reference", "action"]).reference.tolist() + test.drop_duplicates(
            subset=["session_id", "reference", "action"]).reference.tolist()
        unique_clickout_items, counts = np.unique(clickout_item_ids, return_counts=True)

        self.clickout_count_dict = dict(zip(unique_clickout_items, counts))

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

        self.cat_encoders["filters"] = LabelEncoder()
        self.cat_encoders["filters"].fit(filter_set)
        self.continuous_features = [
            "price_diff",
            "price",
            "price_ratio",
        ]

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

            transformed_impressions = self.cat_encoders["reference"].transform(
                row.impressions, to_np=True
            )
            sess_step = row.sess_step
            session_id = row.session_id

            current_session_interactions = session_interactions[session_id][: self.config.sess_length + sess_step - 1]
            current_session_interactions = current_session_interactions[-self.config.sess_length:]

            current_session_actions = session_actions[session_id][: self.config.sess_length + sess_step - 1]
            current_session_actions = current_session_actions[-self.config.sess_length:]

            if row.last_item in transformed_impressions:
                last_interact_index = transformed_impressions.tolist().index(row.last_item)

            label = transformed_impressions == row.reference
            padded_prices = [row.prices[0]] * 2 + row.prices + [row.prices[-1]] * 2
            price_rank = [sorted(row.prices).index(i) for i in row.prices]
            current_rows = np.zeros([len(row.impressions), 21], dtype=object)
            current_rows[:, 0] = row.user_id
            current_rows[:, 1] = transformed_impressions
            current_rows[:, 2] = label
            current_rows[:, 3] = row.session_id
            current_rows[:, 4] = [np.array(self.past_interaction_dict[row.user_id])] * len(row.impressions)
            current_rows[:, 5] = price_rank
            current_rows[:, 6] = row.city
            current_rows[:, 7] = row.last_item
            current_rows[:, 8] = np.arange(len(transformed_impressions))
            current_rows[:, 9] = row.step
            current_rows[:, 10] = row.id
            current_rows[:, 11] = [np.array(current_session_interactions)] * len(row.impressions)
            current_rows[:, 12] = [np.array(current_session_actions)] * len(row.impressions)
            current_rows[:, 13] = row.prices
            current_rows[:, 14] = self.last_click_sess_dict[row.session_id]
            current_rows[:, 15] = self.sess_last_imp_idx_dict[row.session_id]
            current_rows[:, 16] = last_interact_index
            current_rows[:, 17] = row.prices - self.sess_last_price_dict[row.session_id] if self.sess_last_price_dict[
                row.session_id] else 0

            current_rows[:, 18] = row.prices / self.sess_last_price_dict[row.session_id] if self.sess_last_price_dict[
                row.session_id] else 0
            current_rows[:, 19] = [padded_prices[i: i + 5] for i in range(len(row.impressions))]

            current_rows[:, 20] = row.city_platform
            if training or row.reference == self.transformed_nan_item:
                df_list.append(current_rows)
            else:
                label_test_df_list.append(current_rows)
            self.past_interaction_dict[row.user_id] = self.past_interaction_dict[row.user_id][1:]
            self.past_interaction_dict[row.user_id].append(row.reference)

            self.last_click_sess_dict[row.session_id] = row.reference
            self.last_impressions_dict[row.session_id] = transformed_impressions.tolist()

            if row.reference != self.transformed_nan_item:
                self.sess_last_imp_idx_dict[row.session_id] = (transformed_impressions == row.reference).tolist().index(
                    True)
                self.sess_last_price_dict[row.session_id] = (
                    np.array(row.prices)[transformed_impressions == row.reference][0])

        data = np.vstack(df_list)
        dtype_dict = {
            "city": "int32",
            "last_item": "int32",
            "impression_index": "int32",
            "step": "int32",
            "id": "int32",
            "user_id": "int32",
            "reference": "int32",
            "label": "int32",
            "price_rank": "int32",
            "price": "float32",
            "last_click_item": "int32",
            "last_interact_index": "int16",
            "price_diff": "float32",
            "price_ratio": "float32",
            "city_platform": "int32",
        }
        df_columns = [
            "user_id",
            "reference",
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
            "last_interact_index",
            "price_diff",
            "price_ratio",
            "neighbor_prices",
            "city_platform",
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


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.all_cat_columns = self.config.all_cat_columns
        self.categorical_emb_dim = config.categorical_emb_dim
        self.hidden_dims = config.hidden_dims
        self.num_embeddings = config.num_embeddings

        self.emb_dict = torch.nn.ModuleDict()
        for cat_col in self.config.all_cat_columns:
            self.emb_dict[cat_col] = torch.nn.Embedding(
                num_embeddings=self.num_embeddings[cat_col],
                embedding_dim=self.categorical_emb_dim,
            )
        self.gru_sess = torch.nn.GRU(
            input_size=self.categorical_emb_dim * 2,
            hidden_size=self.categorical_emb_dim // 2,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )

        self.hidden1 = torch.nn.Linear(self.categorical_emb_dim * 15, self.hidden_dims[0])
        self.hidden2 = torch.nn.Linear(self.hidden_dims[0] + config.continuous_size * 2 + 3 + config.neighbor_size,
                                       self.hidden_dims[1])
        self.output = torch.nn.Linear(self.hidden_dims[1], 1)
        self.bn = torch.nn.BatchNorm1d(self.categorical_emb_dim * 15)

    def forward(
            self,
            item_id,
            past_interactions,
            mask,
            price_rank,
            city,
            last_item,
            impression_index,
            cont_features,
            past_interactions_sess,
            past_actions_sess,
            last_click_item,
            last_click_impression,
            last_interact_index,
            neighbor_prices,
            city_platform,
    ):
        emb_item = self.emb_dict["item_id"](item_id)
        emb_past_interactions = self.emb_dict["item_id"](past_interactions)
        emb_price_rank = self.emb_dict["price_rank"](price_rank)
        emb_city = self.emb_dict["city"](city)
        emb_last_item = self.emb_dict["item_id"](last_item)
        emb_impression_index = self.emb_dict["impression_index"](impression_index)
        emb_past_interactions_sess = self.emb_dict["item_id"](past_interactions_sess)
        emb_past_actions_sess = self.emb_dict["action"](past_actions_sess)
        emb_last_click_item = self.emb_dict["item_id"](last_click_item)
        emb_last_click_impression = self.emb_dict["impression_index"](
            last_click_impression
        )
        emb_last_interact_index = self.emb_dict["impression_index"](last_interact_index)
        emb_city_platform = self.emb_dict["city_platform"](city_platform)
        emb_past_interactions = emb_past_interactions.permute(0, 2, 1)
        pooled_interaction = F.max_pool1d(
            emb_past_interactions, kernel_size=self.config.sequence_length
        ).squeeze(2)
        emb_past_interactions_sess = torch.cat(
            [emb_past_interactions_sess, emb_past_actions_sess], dim=2
        )
        emb_past_interactions_sess, _ = self.gru_sess(emb_past_interactions_sess)
        emb_past_interactions_sess = emb_past_interactions_sess.permute(0, 2, 1)
        pooled_interaction_sess = F.max_pool1d(emb_past_interactions_sess, kernel_size=self.config.sess_length).squeeze(
            2)
        item_interaction = emb_item * pooled_interaction
        item_last_item = emb_item * emb_last_item
        item_last_click_item = emb_item * emb_last_click_item
        imp_last_idx = emb_impression_index * emb_last_interact_index
        emb_list = [
            emb_item,
            pooled_interaction,
            emb_price_rank,
            emb_city,
            emb_last_item,
            emb_impression_index,
        ]
        emb_concat = torch.cat(emb_list, dim=1)
        sum_squared = torch.pow(torch.sum(emb_concat, dim=1), 2).unsqueeze(1)
        squared_sum = torch.sum(torch.pow(emb_concat, 2), dim=1).unsqueeze(1)
        second_order = 0.5 * (sum_squared - squared_sum)
        squared_cont = torch.pow(cont_features, 2)
        concat = torch.cat(
            [
                emb_item,
                pooled_interaction,
                emb_price_rank,
                emb_city,
                emb_last_item,
                emb_impression_index,
                item_interaction,
                item_last_item,
                pooled_interaction_sess,
                emb_last_click_item,
                emb_last_click_impression,
                emb_last_interact_index,
                item_last_click_item,
                imp_last_idx,
                emb_city_platform,
            ],
            dim=1,
        )
        concat = self.bn(concat)
        hidden = torch.nn.ReLU()(self.hidden1(concat))
        hidden = torch.cat(
            [
                cont_features,
                hidden,
                sum_squared,
                squared_sum,
                second_order,
                squared_cont,
                neighbor_prices,
            ],
            dim=1,
        )
        hidden = self.bn(hidden)
        hidden = torch.nn.ReLU()(self.hidden2(hidden))
        return torch.sigmoid(self.output(hidden)).squeeze()


class Configuration:
    def __init__(self):
        self.num_epochs = 1
        self.batch_size = 1024
        self.loss = torch.nn.BCELoss
        self.categorical_emb_dim = 256
        self.learning_rate = 0.01
        self.weight_decay = 0
        self.sequence_length = 10
        self.sess_length = 30
        self.num_embeddings = {}
        self.hidden_dims = [256, 128]

    def __getitem__(self, x):
        return getattr(self, x)

    def __setitem__(self, x, v):
        return setattr(self, x, v)


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_adam_optimizer(network, params):
    return torch.optim.Adam(
        network.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )


def compute_mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def get_prediction(loader, net, loss_function):
    net.eval()
    all_scores = []
    validation_loss = []
    for data in loader:
        with torch.no_grad():
            prediction = net(*[i for idx, i in enumerate(data) if idx != 1])
            targets = data[1]
            loss = loss_function(prediction, targets).item()
            prediction = prediction.detach().cpu().numpy().tolist()
            all_scores += prediction
            validation_loss.append(loss)
    validation_loss = np.mean(validation_loss)
    return all_scores, validation_loss


def evaluate_valid(val_loader, val_df, net, loss_function):
    val_df['score'], val_loss = get_prediction(val_loader, net, loss_function)
    grouped_val = val_df.groupby('session_id')
    rss = []
    for session_id, group in grouped_val:
        scores = group['score']
        sorted_arg = np.flip(np.argsort(scores))
        rss.append(group['label'].values[sorted_arg])

    return compute_mean_reciprocal_rank(rss)


def run():
    seed_everything(10)
    configuration = Configuration()
    data_gen = NNDataGenerator(configuration)
    valid_data = data_gen.val_data
    loss_function = configuration.loss()
    net = Net(configuration)
    optimizer = get_adam_optimizer(net, configuration)
    val_loader = data_gen.evaluate_data_valid()
    train_loader = data_gen.instance_a_train_loader()
    for i in range(configuration.num_epochs):
        net.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            prediction = net(*[i for idx, i in enumerate(data) if idx != 1])
            targets = data[1]
            loss = loss_function(prediction, targets)
            loss.backward()
            optimizer.step()

        return evaluate_valid(val_loader, valid_data, net, loss_function)


if __name__ == "__main__":
    print("mrr", run())
