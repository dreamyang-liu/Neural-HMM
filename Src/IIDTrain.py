import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from Core.Util import get_results, anno_space_map


def log_entropy_loss(x, dim=-1):
    # x is in log-domain
    h = torch.exp(x) * x
    h = -1.0 * h.sum(dim=dim)
    return h


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.n_src = args.n_src
        self.device = args.device

    def pre_train(self, data_loader, optimizer, prior_, emiss_):
        train_loss = 0
        num_samples = 0

        self.model.nn_module.train()
        if prior_ is not None:
            prior_ = prior_.to(self.device)
        if emiss_ is not None:
            emiss_ = emiss_.to(self.device)

        for i, batch in enumerate(tqdm(data_loader)):
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_hidden_states, nn_emiss = self.model.nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden = nn_hidden_states.size()
            n_obs = nn_emiss.size(-1)
            if prior_ is not None:
                l1 = F.mse_loss(
                    nn_hidden_states, prior_.view(1, 1, n_hidden).repeat(batch_size, max_seq_len, 1)
                )
            else:
                l1 = 0
            if emiss_ is not None:
                l2 = F.mse_loss(
                    nn_emiss, emiss_.view(1, 1, self.n_src, n_hidden, n_obs).repeat(batch_size, max_seq_len, 1, 1, 1)
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
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            # training step
            optimizer.zero_grad()
            log_probs, _ = self.model(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens, normalize_observation=self.args.obs_normalization
            )

            loss = -log_probs.mean()
            loss.backward()
            optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        # print(train_loss)

        return train_loss

    def test(self, data_loader):

        self.model.eval()
        batch_pred_span = list()
        batch_true_span = list()
        batch_sent = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])

                # get prediction
                pred_span, _ = self.model.annotate(
                    emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens, label_set=self.args.bio_lbs,
                    normalize_observation=self.args.obs_normalization
                )
                if hasattr(self.args, 'mappings'):
                    if self.args.mappings is not None:
                        pred_span = [anno_space_map(ps, self.args.mappings, self.args.lbs) for ps in pred_span]
                batch_pred_span += pred_span

                # Save source text and spans
                batch_sent += batch[-2]
                batch_true_span += batch[-1]
            results = get_results(batch_pred_span, batch_true_span, batch_sent, all_labels=self.args.lbs)
        return results
