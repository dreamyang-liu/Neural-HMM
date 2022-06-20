import torch
import numpy as np

from typing import List, Optional
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: List[str],
                 embs: List[torch.Tensor],
                 obs: List[List[int]],
                 lbs: List[List[str]]
                 ):
        """
        A wrapper class to create syntax dataset for syntax expansion training.
        """
        super().__init__()
        self._embs = embs
        self._obs = obs
        self._text = text
        self._lbs = lbs

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._obs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._text[idx], self._embs[idx], self._obs[idx], self._lbs[idx]


def batch_prep(emb_list: List[torch.Tensor],
               obs_list: List[torch.Tensor],
               txt_list: Optional[List[List[str]]] = None,
               lbs_list: Optional[List[List[str]]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    for emb, obs, txt, lbs in zip(emb_list, obs_list, txt_list, lbs_list):
        assert len(obs) + 1 == len(emb) == len(txt) == len(lbs)
    d_emb = emb_list[0].size(-1)
    _, n_src, n_obs = obs_list[0].size()
    seq_lens = [len(obs)+1 for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    emb_batch = torch.stack([
        torch.cat([inst, torch.zeros([max_seq_len-len(inst), d_emb])], dim=-2) for inst in emb_list
    ])

    prefix = torch.zeros([1, n_src, n_obs])
    prefix[:, :, 0] = 1
    obs_batch = torch.stack([
        torch.cat([prefix.clone(), inst, prefix.repeat([max_seq_len-len(inst)-1, 1, 1])])
        for inst in obs_list
    ])
    obs_batch /= obs_batch.sum(dim=-1, keepdim=True)

    # obs_batch = torch.tensor(obs_batch, dtype=torch.float)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    return emb_batch, obs_batch, seq_lens, txt_list, lbs_list


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    txt, embs, obs, lbs = list(zip(*insts))
    batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt, lbs_list=lbs)
    return batch
