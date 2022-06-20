import torch
import numpy as np
from torch.nn import functional as F
from tqdm.auto import tqdm
from Core.Util import one_hot_to_string


def log_entropy_loss(x, dim=-1):
    # x is in log-domain
    h = torch.exp(x) * x
    h = -1.0 * h.sum(dim=dim)
    return h


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device

    def pre_train(self, data_loader, optimizer, trans_, emiss_):
        train_loss = 0
        num_samples = 0

        self.model.nn_module.train()
        if trans_ is not None:
            trans_ = trans_.to(self.device)
        if emiss_ is not None:
            emiss_ = emiss_.to(self.device)

        for i, batch in enumerate(tqdm(data_loader)):
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch)
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_trans, nn_emiss = self.model.nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()
            n_obs = nn_emiss.size(-1)
            if trans_ is not None:
                l1 = F.mse_loss(
                    nn_trans, trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)
                )
            else:
                l1 = 0
            if emiss_ is not None:
                l2 = F.mse_loss(
                    nn_emiss, emiss_.view(1, 1, n_hidden, n_obs).repeat(batch_size, max_seq_len, 1, 1)
                )
            else:
                l2 = 0
            loss = l1 + l2
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        return train_loss

    def train(self, data_loader, optimizer):
        train_loss = 0
        num_samples = 0

        self.model.train()

        for i, batch in enumerate(tqdm(data_loader)):
            # get data
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch)
            batch_size = len(obs_batch)
            num_samples += batch_size

            # training step
            optimizer.zero_grad()
            log_probs, (log_trans, log_emiss) = self.model(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens
            )

            trans_hloss = log_entropy_loss(log_trans).mean(dim=1)
            emiss_hloss = log_entropy_loss(log_emiss).mean(dim=1)

            # TODO: the coefficients are subject to change
            loss = -log_probs.mean() - 0.5 * (1/(np.exp(i-2)+1)) * trans_hloss.mean() \
                   - 0.2 * (1/(np.exp(i-2)+1)) * emiss_hloss.mean()
            loss = -log_probs.mean() - 0.5 * trans_hloss.mean() - 0.2 * emiss_hloss.mean()
            loss.backward()
            optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        # print(train_loss)

        return train_loss

    def test(self, data_loader, idx2label):

        self.model.eval()
        accuracy = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch)

                # inference
                batch_label_indices, batch_probs = self.model.inference(
                    emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens
                )

                # index to labels
                batch_labels = [[idx2label[lb_index] for lb_index in label_indices]
                                for label_indices in batch_label_indices]
                batch_obs = [[idx2label[lb_index.item()] for lb_index in one_hot_to_string(obs)[:length]]
                             for obs, length in zip(obs_batch, seq_lens)]
                # print(labels)
                # print(true_obs)
                # print()

                # calculate acucracies
                accuracy += np.sum([(np.array(lbs) == np.array(obs)).sum() / len(lbs)
                                    for lbs, obs in zip(batch_labels, batch_obs)]) / len(batch_obs)
            mean_accu = accuracy / (i+1)
        return mean_accu
