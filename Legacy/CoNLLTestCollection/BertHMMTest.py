import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from argparse import Namespace
from Legacy.CoNLL03.SingleObsModel import NeuralHMM
from Legacy.CoNLL03.SingleObsTrain import Trainer
from typing import List
from Core.Util import load_labels, one_hot,set_seed_everywhere, plot_tran_emis, plot_train_results


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 embs: List[torch.Tensor],
                 obs: List[List[int]]
                 ):
        """
        A wrapper class to create syntax dataset for syntax expansion training.
        """
        super().__init__()
        self._embs = embs
        self._obs = obs

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._obs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._embs[idx], self._obs[idx]


def batch_prep(emb_list: List[torch.Tensor], obs_list: List[List[int]], n_state: int):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    d_emb = emb_list[0].size(-1)
    seq_lens = [len(obs) for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    emb_batch = torch.stack([
        torch.cat([inst, torch.zeros([max_seq_len-len(inst), d_emb])], dim=-2) for inst in emb_list
    ])

    obs_batch = np.array([
        inst + [0] * (max_seq_len - len(inst)) for inst in obs_list
    ])
    obs_batch = one_hot(obs_batch, n_state)

    # obs_batch = torch.tensor(obs_batch, dtype=torch.float)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    return emb_batch, obs_batch, seq_lens


def collate_fn(insts, n_state):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    embs, obs = list(zip(*insts))
    batch = batch_prep(embs, obs_list=obs, n_state=n_state)
    return batch


def main():
    args = Namespace(
        data_dir=r'../../data',
        data_name='bert-data-toy.pt',
        batch_size=256,
        d_emb=768,
        dropout=0.1,
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
    idx2label = load_labels(os.path.join('../../data', 'CoNLL2003-labels.json'))
    label2idx = {v: k for k, v in enumerate(idx2label)}

    # construct dataset
    embs = data['embs']
    lb_indices = data['label-indices']
    data_set = Dataset(embs=embs, obs=lb_indices)
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

    # emis_matrix = torch.eye(args.n_hidden)
    # emis_matrix = emis_matrix + (torch.rand([args.n_hidden, args.n_hidden]) - 0.5) / 2
    # emis_matrix = emis_matrix / emis_matrix.sum(dim=1, keepdim=True)
    emis_matrix = None

    tr_matrix = torch.zeros([len(label2idx), len(label2idx)])
    for lb_index in lb_indices:
        for l0, l1 in zip(lb_index[:-1], lb_index[1:]):
            tr_matrix[l0, l1] += 1
    tr_matrix = tr_matrix / tr_matrix.sum(dim=1, keepdim=True)
    # tr_matrix = tr_matrix + (torch.rand([args.n_hidden, args.n_hidden]) - 0.5) / 2
    # tr_matrix = tr_matrix / tr_matrix.sum(dim=1, keepdim=True)
    # tr_matrix = None

    # about saving checkpoint
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_dir = os.path.join(args.model_dir, 'model.best.chkpt')
    if args.debugging_mode and not os.path.isdir(args.figure_dir):
        os.mkdir(args.figure_dir)

    # if args.debugging_mode:
    #     plot_tran_emis(
    #         os.path.join(args.figure_dir, f'initial_matrix.png'),
    #         tr_matrix,
    #         emis_matrix
    #     )

    # initialize model
    hmm_model = NeuralHMM(args=args, state_prior=state_prior, trans_matrix=tr_matrix, emiss_matrix=emis_matrix)
    hmm_model.to(device=args.device)

    # initialize optimizer
    hmm_params = [
        hmm_model.unnormalized_emiss,
        hmm_model.unnormalized_trans,
        hmm_model.state_priors
    ]
    optimizer = torch.optim.Adam([
        {'params': hmm_model.nn_module.parameters(), 'lr': 1e-4},
        {'params': hmm_params}
    ],
        lr=args.lr,
        weight_decay=1e-5
    )

    pre_train_optimizer = torch.optim.Adam(
        hmm_model.nn_module.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    # initialize training process
    trainer = Trainer(hmm_model, args)
    min_loss = np.inf
    stop_count = 0
    accu_list = []

    # pre-train neural module
    print("[INFO] pre-training neural module")
    for epoch_i in range(args.epoch // 2):
        train_loss = trainer.pre_train(data_loader, pre_train_optimizer, tr_matrix, emis_matrix)
        print(f"[INFO] Epoch: {epoch_i}, Loss: {train_loss}")

    # start training process
    for epoch_i in range(args.epoch):
        print("========= Epoch %d of %d =========" % (epoch_i + 1, args.epoch))
        train_loss = trainer.train(data_loader, optimizer)
        mean_accu = trainer.test(data_loader, idx2label)

        print("========= Results: epoch %d of %d =========" % (epoch_i + 1, args.epoch))
        print("[INFO] train loss: %.4f" % train_loss)
        print("[INFO] test accuracy: %.4f" % mean_accu)

        # check convergence
        if min_loss - train_loss < train_loss * 1E-3:
            stop_count += 1
        else:
            min_loss = train_loss
            stop_count = 0

        accu_list.append(mean_accu)
        # save model
        model_state_dict = hmm_model.state_dict()
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
                os.path.join(args.figure_dir, 'matrices-bert.{:2d}.png'.format(epoch_i)),
                torch.softmax(hmm_model.unnormalized_trans, dim=-1).detach().cpu().numpy(),
                torch.softmax(hmm_model.unnormalized_emiss, dim=-1).detach().cpu().numpy()
            )
    if args.debugging_mode:
        plot_train_results(
            os.path.join(args.figure_dir, 'matrices-bert-accu.png'),
            accu_list
        )


if __name__ == '__main__':
    main()
