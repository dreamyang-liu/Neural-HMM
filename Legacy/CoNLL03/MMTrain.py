import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from Core.Data import label_to_span
from Core.Constants import CoNLL_BIO
from Core.Util import get_results


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device
        self.n_hidden = args.n_hidden

    def pre_train(self, data_loader, optimizer, trans_):
        train_loss = 0
        num_samples = 0

        self.model.nn_module.train()
        if trans_ is not None:
            trans_ = trans_.to(self.device)

        for i, batch in enumerate(tqdm(data_loader)):
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_trans = self.model.nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()

            loss_mask = torch.zeros([batch_size, max_seq_len], device=self.device)
            for i in range(batch_size):
                loss_mask[i, :seq_lens[i]] = 1
            pred = loss_mask.view(batch_size, max_seq_len, 1, 1) * nn_trans
            true = loss_mask.view(batch_size, max_seq_len, 1, 1) * \
                   trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)
            if trans_ is not None:
                l1 = F.mse_loss(pred, true)
            else:
                l1 = 0
            loss = l1
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
            log_probs = self.model(emb=emb_batch)

            # loss = F.cross_entropy(
            #     log_probs.view(-1, self.n_hidden), obs_batch.view(-1), ignore_index=-1
            # )
            loss = F.nll_loss(log_probs.view(-1, self.n_hidden), obs_batch.view(-1), ignore_index=-1)

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
                log_state_probs = self.model(emb=emb_batch)
                pred_idx = torch.argmax(log_state_probs, dim=-1).cpu().numpy()
                pred_lbs = list()
                for indices, lengths in zip(pred_idx, seq_lens):
                    pred_lbs.append([CoNLL_BIO[idx] for idx in indices[:lengths]])
                pred_span = [label_to_span(pred) for pred in pred_lbs]

                # Construct true spans from labels
                obs_batch = obs_batch.cpu().numpy()
                true_lbs = list()
                for indices, lengths in zip(obs_batch, seq_lens):
                    true_lbs.append([CoNLL_BIO[idx] for idx in indices[:lengths]])
                true_span = [label_to_span(lbs) for lbs in true_lbs]

                # Save source text and spans
                batch_sent += batch[-1]
                batch_pred_span += pred_span
                batch_true_span += true_span
            results = get_results(batch_pred_span, batch_true_span, batch_sent)
        return results
