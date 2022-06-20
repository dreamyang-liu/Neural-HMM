import torch
import numpy as np
from typing import List, Optional
from torch.utils.data import DataLoader
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


def batch_prep(obs_list: List[List[int]]):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    seq_lens = [len(obs) for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    obs_batch = np.array([
        inst + [0] * (max_seq_len - len(inst))
        for inst in obs_list
    ])

    obs_batch = torch.tensor(obs_batch, dtype=torch.long)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    return obs_batch, seq_lens


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    batch = batch_prep(insts)
    return batch
