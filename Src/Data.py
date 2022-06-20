import torch
import numpy as np

from typing import List, Optional
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: List[str],
                 embs: List[torch.Tensor],
                 obs: List[List[int]],
                 lbs: Optional[List[List[str]]] = None
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
        if self._lbs is not None:
            return self._text[idx], self._embs[idx], self._obs[idx], self._lbs[idx]
        else:
            return self._text[idx], self._embs[idx], self._obs[idx]


def batch_prep(emb_list: List[torch.Tensor],
               obs_list: List[torch.Tensor],
               txt_list: Optional[List[List[str]]] = None,
               lbs_list: Optional[List[dict]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    for emb, obs, txt in zip(emb_list, obs_list, txt_list):
        assert len(obs) + 1 == len(emb) == len(txt)
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

    # increment the indices of the true spans
    lbs_batch = [{(i+1, j+1): v for (i, j), v in lbs.items()} for lbs in lbs_list] \
        if lbs_list is not None else None

    # obs_batch = torch.tensor(obs_batch, dtype=torch.float)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    return emb_batch, obs_batch, seq_lens, txt_list, lbs_batch


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    all_insts = list(zip(*insts))
    if len(all_insts) == 4:
        txt, embs, obs, lbs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt, lbs_list=lbs)
    elif len(all_insts) == 3:
        txt, embs, obs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt)
    else:
        raise ValueError
    return batch


def annotate_data(model, text, embs, obs, lbs, args):
    dataset = Dataset(text=text, embs=embs, obs=obs, lbs=lbs)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    
    model.eval()
    score_list = list()
    span_list = list()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            # get data
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(args.device), batch[:3])
            # get prediction
            # the scores are shifted back, i.e., len = len(emb)-1 = len(sentence)
            _, (scored_spans, scores) = model.annotate(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens, label_set=args.bio_lbs,
                normalize_observation=args.obs_normalization
            )
            score_list += scores
            span_list += scored_spans
    return span_list, score_list


def update_src_data(model, src_data, model_name, save_dir, text, embs, obs, lbs, args):
    pred_spans = annotate_data(model, text=text, embs=embs, obs=obs, lbs=lbs, args=args)
    data_sents = src_data['sentences']
    data_annotations = src_data['annotations']
    data_lbs = src_data['labels']
    results = src_data['results'] if "results" in src_data.keys() else dict()
    results[model_name] = pred_spans
    data = {
        "sentences": data_sents,
        "annotations": data_annotations,
        "labels": data_lbs,
        "results": results
    }
    torch.save(data, save_dir)
