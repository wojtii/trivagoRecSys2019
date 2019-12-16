from tqdm import tqdm

from data import NNDataGenerator
from nn import Net

import torch
import os
import random
import numpy as np


class Configuration:
    def __init__(self):
        self.num_epochs = 1
        self.batch_size = 1024
        self.loss = torch.nn.BCELoss
        self.categorical_emb_dim = 128
        self.learning_rate = 0.001
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
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_adam_optimizer(network, params):
    return torch.optim.Adam(
        network.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        eps=1e-07,
        amsgrad=True,
    )


def compute_mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def get_prediction(loader, net, crit):
    net.eval()
    all_scores = []
    validation_loss = []
    for data in loader:
        with torch.no_grad():
            prediction = net(*[i for idx, i in enumerate(data) if idx != 1])
            targets = data[1]
            loss = crit(prediction, targets).item()
            prediction = prediction.detach().cpu().numpy().tolist()
            all_scores += prediction
            validation_loss.append(loss)
    validation_loss = np.mean(validation_loss)
    return all_scores, validation_loss


def evaluate_valid(val_loader, val_df, net, crit):
    val_df['score'], val_loss = get_prediction(val_loader, net, crit)
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
    crit = configuration.loss()
    net = Net(configuration)
    optim = get_adam_optimizer(net, configuration)
    val_loader = data_gen.evaluate_data_valid()
    train_loader = data_gen.instance_a_train_loader()
    for i in range(configuration.num_epochs):
        net.train()
        for data in tqdm(train_loader):
            optim.zero_grad()
            prediction = net(*[i for idx, i in enumerate(data) if idx != 1])
            targets = data[1]
            loss = crit(prediction, targets)
            loss.backward()
            optim.step()

        return evaluate_valid(val_loader, valid_data, net, crit)


if __name__ == "__main__":
    print("mrr", run())
