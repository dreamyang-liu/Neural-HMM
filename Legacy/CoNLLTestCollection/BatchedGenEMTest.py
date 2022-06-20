import os
import torch
import numpy as np
from Core.Util import load_labels, one_hot,\
    set_seed_everywhere, one_hot_to_string, plot_train_results, plot_tran_emis
from typing import List
from torch.utils.data import DataLoader
from argparse import Namespace
from Legacy.BatchedGenEM import NeuralHMM
from tqdm.auto import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 obs: List[List[int]]
                 ):
        """
        A wrapper class to create syntax dataset for syntax expansion training.
        """
        super().__init__()
        self._obs = obs

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._obs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._obs[idx]


def batch_prep(obs_list: List[List[int]], n_state: int):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    seq_lens = [len(obs) for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    obs_batch = np.array([
        inst + [0] * (max_seq_len - len(inst))
        for inst in obs_list
    ])
    obs_batch = one_hot(obs_batch, n_state)

    # obs_batch = torch.tensor(obs_batch, dtype=torch.float)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    return obs_batch, seq_lens


def collate_fn(insts, n_state):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    batch = batch_prep(insts, n_state=n_state)
    return batch


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device

    def train(self, data_loader, optimizer):

        train_loss = 0
        num_samples = 0

        self.model.train()

        for i, batch in enumerate(tqdm(data_loader)):
            # get data
            obs_batch, seq_lens = map(lambda x: x.to(self.device), batch)
            batch_size = len(obs_batch)
            num_samples += batch_size

            # training step
            optimizer.zero_grad()
            log_probs = self.model(obs=obs_batch, seq_lengths=seq_lens)

            loss = -log_probs.mean()
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
                obs_batch, seq_lens = map(lambda x: x.to(self.device), batch)
                batch_label_indices, batch_probs = self.model.inference(obs=obs_batch, seq_lengths=seq_lens)
                batch_labels = [[idx2label[lb_index] for lb_index in label_indices]
                                for label_indices in batch_label_indices]
                batch_obs = [[idx2label[lb_index.item()] for lb_index in one_hot_to_string(obs)[:length]]
                             for obs, length in zip(obs_batch, seq_lens)]
                # print(labels)
                # print(true_obs)
                # print()
                accuracy += np.sum([(np.array(lbs) == np.array(obs)).sum() / len(lbs)
                                    for lbs, obs in zip(batch_labels, batch_obs)]) / len(batch_obs)
            mean_accu = accuracy / (i+1)
        return mean_accu


def main():
    args = Namespace(
        data_dir=r'../../data/',
        data_name='bert-data.pt',
        batch_size=1024,
        d_emb=768,
        dropout=0.05,
        epoch=20,
        lr=0.1,
        model_dir='../../models',
        n_hidden=9,
        n_obs=9,
        num_workers=0,
        pin_memory=False,
        random_seed=42,
        debugging_mode=True,
        figure_dir='../../plots',
        test_size=0.1,
        device=torch.device('cuda')
    )
    set_seed_everywhere(args.random_seed)
    if args.debugging_mode:
        torch.autograd.set_detect_anomaly(True)

    # load data
    data = torch.load(os.path.join(args.data_dir, args.data_name))
    lb_indices = data['label-indices']
    # all_sentences, all_labels = load_conll_2003_data(os.path.join(data_dir, data_name))
    idx2label = load_labels(os.path.join('../../data', 'CoNLL2003-labels.json'))
    label2idx = {v: k for k, v in enumerate(idx2label)}

    data_set = Dataset(obs=lb_indices)
    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=args.batch_size,
        collate_fn=lambda x: collate_fn(x, n_state=args.n_obs),
        shuffle=True,
        pin_memory=False,
        drop_last=False
    )

    # inject prior knowledge about transition and emission
    state_prior = torch.zeros(args.n_hidden, device=args.device) + 1e-2
    state_prior[0] += 1 - state_prior.sum()

    emis_matrix = torch.eye(args.n_hidden)
    emis_matrix = emis_matrix + (torch.rand([args.n_hidden, args.n_hidden]) - 0.5) / 2
    emis_matrix = emis_matrix / emis_matrix.sum(dim=1, keepdim=True)
    # emis_matrix = None

    tr_matrix = torch.zeros([len(label2idx), len(label2idx)])
    for lb_index in lb_indices:
        for l0, l1 in zip(lb_index[:-1], lb_index[1:]):
            tr_matrix[l0, l1] += 1
    tr_matrix = tr_matrix / tr_matrix.sum(dim=1, keepdim=True)
    tr_matrix = tr_matrix + (torch.rand([args.n_hidden, args.n_hidden]) - 0.5) / 2
    tr_matrix = tr_matrix / tr_matrix.sum(dim=1, keepdim=True)
    # tr_matrix = None

    # about saving checkpoint
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_dir = os.path.join(args.model_dir, 'model.best.chkpt')
    if args.debugging_mode and not os.path.isdir(args.figure_dir):
        os.mkdir(args.figure_dir)

    # initialize model
    hmm_gen_em_model = NeuralHMM(args=args, state_prior=state_prior, trans_matrix=tr_matrix, emiss_matrix=emis_matrix)
    hmm_gen_em_model.to(device=args.device)

    # initialize optimizer
    optimizer = torch.optim.Adam(
        hmm_gen_em_model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    # initialize training process
    trainer = Trainer(hmm_gen_em_model, args)
    min_loss = np.inf
    stop_count = 0
    accu_list = list()

    # start training process
    for epoch_i in range(args.epoch):
        print("========= Epoch %d of %d =========" % (epoch_i + 1, args.epoch))
        train_loss = trainer.train(data_loader, optimizer)
        mean_accu = trainer.test(data_loader, idx2label)

        print("========= Results: epoch %d of %d =========" % (epoch_i + 1, args.epoch))
        print("[INFO] train loss: %.4f" % train_loss)
        print("[INFO] test accuracy: %.4f" % mean_accu)

        # check convergence
        if min_loss - train_loss < train_loss * 1e-4:
            stop_count += 1
        else:
            min_loss = train_loss
            stop_count = 0

        # save model
        accu_list.append(mean_accu)
        model_state_dict = hmm_gen_em_model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'settings': args,
            'epoch': epoch_i
        }
        if stop_count == 0:
            torch.save(checkpoint, model_dir)
            print("[INFO] Checkpoint Saved!\n")
        elif stop_count == 3:
            print("[WARNING] Loss is not decreasing. Exiting program...")
            break

        if args.debugging_mode:
            plot_tran_emis(
                os.path.join(args.figure_dir, f'matrices.{epoch_i}.png'),
                torch.softmax(hmm_gen_em_model.unnormalized_trans, dim=-1).detach().cpu().numpy(),
                torch.softmax(hmm_gen_em_model.unnormalized_emiss, dim=-1).detach().cpu().numpy()
            )
    if args.debugging_mode:
        plot_train_results(
            os.path.join(args.figure_dir, 'matrices-accu.png'),
            accu_list
        )


if __name__ == '__main__':
    main()
