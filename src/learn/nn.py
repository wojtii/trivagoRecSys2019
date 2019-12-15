import torch
import torch.nn.functional as F


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
            if cat_col == "item_id":
                self.emb_dict[cat_col] = torch.nn.Embedding(
                    num_embeddings=self.num_embeddings[cat_col],
                    embedding_dim=self.categorical_emb_dim,
                    padding_idx=self.config.transformed_dummy_item,
                )
            else:
                self.emb_dict[cat_col] = torch.nn.Embedding(
                    num_embeddings=self.num_embeddings[cat_col],
                    embedding_dim=self.categorical_emb_dim,
                )
        # gru for extracting session and user interest
        self.gru_sess = torch.nn.GRU(
            input_size=self.categorical_emb_dim * 2,
            hidden_size=self.categorical_emb_dim // 2,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )
        self.other_item_gru = torch.nn.GRU(
            input_size=self.categorical_emb_dim,
            hidden_size=self.categorical_emb_dim // 2,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )

        # linear layer on top of continuous features
        self.cont_linear = torch.nn.Linear(
            config.continuous_size, self.categorical_emb_dim
        )

        # hidden layerrs
        self.hidden1 = torch.nn.Linear(
            self.categorical_emb_dim * 15, self.hidden_dims[0]
        )
        self.hidden2 = torch.nn.Linear(
            self.hidden_dims[0] + config.continuous_size * 2 + 3 + config.neighbor_size,
            self.hidden_dims[1],
        )

        # output layer
        self.output = torch.nn.Linear(self.hidden_dims[1], 1)

        # batch normalization
        self.bn = torch.nn.BatchNorm1d(self.categorical_emb_dim * 15)
        self.bn_hidden = torch.nn.BatchNorm1d(
            self.hidden_dims[0] + config.continuous_size * 2 + 3 + config.neighbor_size
        )

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
        pooled_interaction_sess = F.max_pool1d(
            emb_past_interactions_sess, kernel_size=self.config.sess_length
        ).squeeze(2)

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

        hidden = self.bn_hidden(hidden)
        hidden = torch.nn.ReLU()(self.hidden2(hidden))

        return torch.sigmoid(self.output(hidden)).squeeze()
