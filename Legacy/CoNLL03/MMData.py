import torch
import numpy as np

from typing import List, Optional
from torch.utils.data import DataLoader
from Core.Constants import CoNLL_BIO
from Core.Util import one_hot


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: List[str],
                 embs: List[torch.Tensor],
                 lbs: List[List[str]]
                 ):
        """
        A wrapper class to create syntax dataset for syntax expansion training.
        """
        super().__init__()
        self._embs = embs
        self._text = text
        self._lbs = lbs

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._text)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._text[idx], self._embs[idx], self._lbs[idx]


def batch_prep(emb_list: List[torch.Tensor],
               lbs_list: Optional[List[List[str]]],
               txt_list: Optional[List[List[str]]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    for emb, txt, lbs in zip(emb_list, txt_list, lbs_list):
        assert len(emb) == len(txt) == len(lbs)
    d_emb = emb_list[0].size(-1)
    seq_lens = [len(emb) for emb in emb_list]
    max_seq_len = np.max(seq_lens)

    emb_batch = torch.stack([
        torch.cat([inst, torch.zeros([max_seq_len-len(inst), d_emb])], dim=-2) for inst in emb_list
    ])

    lbs_batch = np.array([
        inst + [-1] * (max_seq_len - len(inst))
        for inst in lbs_list
    ])

    lbs_batch = torch.tensor(lbs_batch, dtype=torch.long)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    return emb_batch, lbs_batch, seq_lens, txt_list


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    txt, embs, lbs = list(zip(*insts))
    batch = batch_prep(emb_list=embs, lbs_list=lbs, txt_list=txt)
    return batch
