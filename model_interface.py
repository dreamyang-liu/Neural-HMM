import os
import argparse
import torch
import json

import numpy as np
from Core.Util import set_seed_everywhere
from Core.Data import (
    txt_to_token_span,
    build_bert_emb,
    extract_sequence,
    initialise_transmat,
    initialise_emissions,
    token_to_txt_span
)
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from Src.NHMMModel import NeuralHMM
from Src.NHMMTrain import Trainer
from Src.Data import Dataset, collate_fn, annotate_data


def parse_args():
    """
    Wrapper function for parsing arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='',
        help='Indicates which dataset to use. Leave it empty if you are not using prepared data'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='data location.'
    )
    parser.add_argument(
        '--config_dir', type=str, required=True,
        help='The direction of the configuration json file'
    )
    parser.add_argument(
        '--model_dir', type=str, default='',
        help='where to save and load checkpoints.'
    )
    parser.add_argument(
        '--bert_tokenizer', type=str, default='bert-base-uncased',
        help='Which kind of bert tokenizer to use.'
    )
    parser.add_argument(
        '--bert_model', type=str, default='bert-base-uncased',
        help='Which kind of bert model to use.'
    )
    parser.add_argument(
        '--trans_nn_weight', type=float, default=0.5, help='the weight of neural part in the transition matrix'
    )
    parser.add_argument(
        '--emiss_nn_weight', type=float, default=0.5, help='the weight of neural part in the emission matrix'
    )
    parser.add_argument(
        '--epoch', type=int, default=15, help='number of training epochs'
    )
    parser.add_argument(
        '--pretrain_epoch', type=int, default=5, help='number of training epochs'
    )
    parser.add_argument(
        '--hmm_lr', type=float, default=0.01, help='learning rate'
    )
    parser.add_argument(
        '--nn_lr', type=float, default=0.001, help='learning rate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout ratio'
    )
    parser.add_argument(
        '--no_cuda', action='store_true', help='disable cuda'
    )
    parser.add_argument(
        '--obs_normalization', type=bool, default=True,
        help='normalizing observations'
    )
    parser.add_argument(
        '--pin_memory', action='store_true', help='whether pin your cuda memory during training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help='how many processes to use when loading data'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args


def build_config(args):
    with open(args.config_dir, 'r') as f:
        config_file = json.load(f)

    args.lbs = config_file['lbs']
    args.src_to_keep = config_file['src_to_keep']
    try:
        args.src_priors = config_file['src_priors']
    except KeyError:
        args.src_priors = {src: {lb: (0.8, 0.8) for lb in args.lbs} for src in args.src_to_keep}
    args.bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in args.lbs for bi in "BI"]
    args.lbs2idx = {label: i for i, label in enumerate(args.bio_lbs)}

    return args


def build_dataset(args):
    # load data from disk
    # TODO: I don't know in what format the data would be stored, so I assume it is json.
    print("[INFO] Loading data...")
    with open(args.data_dir, 'r') as f:
        data = json.load(f)

    # convert data format
    sent_list = list()
    weak_anno_list = list()
    for inst in data['processedData']:
        sent_list.append(inst['sentence'].strip())
        src_anno_dict = dict()
        for src in args.src_to_keep:
            scored_spans = dict()
            for anno in inst['annotation']:
                if anno[0] == src:
                    scored_spans[(anno[2], anno[3])] = [(anno[1], 1.0)]
            src_anno_dict[src] = scored_spans
        weak_anno_list.append(src_anno_dict)

    # convert character-space annotations to token-level
    token_list = [word_tokenize(sent) for sent in sent_list]
    assert len(weak_anno_list) == len(weak_anno_list)
    token_anno_list = list()
    for tokens, sent, weak_anno in zip(token_list, sent_list, weak_anno_list):
        anno_src_dict = dict()
        for src, spans in weak_anno.items():
            token_spans = txt_to_token_span(tokens, sent, spans)
            token_annos = dict()
            for v, span in zip(spans.values(), token_spans):
                token_annos[span] = v
            anno_src_dict[src] = token_annos
        token_anno_list.append(anno_src_dict)

    print("[INFO] Data loaded!")

    return sent_list, token_list, token_anno_list


# noinspection PyTypeChecker
def build_embeddings(token_list, args):

    print("[INFO] Building Embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
    model = AutoModel.from_pretrained(args.bert_model).to(args.device)

    standarized_sents = list()
    o2n_map = list()
    n = 0
    for i, sents in enumerate(token_list):
        nltk_string = ' '.join(sents)
        len_bert_tokens = len(tokenizer.tokenize(nltk_string))

        if len_bert_tokens >= 510:
            nltk_tokens_list = [sents]
            bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in nltk_tokens_list]
            while (np.asarray(bert_length_list) >= 510).any():
                new_list = list()
                for nltk_tokens, bert_len in zip(nltk_tokens_list, bert_length_list):

                    if bert_len < 510:
                        new_list.append(nltk_tokens)
                        continue

                    nltk_string = ' '.join(nltk_tokens)
                    sts = sent_tokenize(nltk_string)

                    sent_lens = list()
                    for st in sts:
                        sent_lens.append(len(word_tokenize(st)))
                    ends = [np.sum(sent_lens[:i]) for i in range(1, len(sent_lens) + 1)]

                    nearest_end_idx = np.argmin((np.array(ends) - len(nltk_tokens) / 2) ** 2)
                    split_1 = nltk_tokens[:ends[nearest_end_idx]]
                    split_2 = nltk_tokens[ends[nearest_end_idx]:]
                    new_list.append(split_1)
                    new_list.append(split_2)
                nltk_tokens_list = new_list
                bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in nltk_tokens_list]
            n_splits = len(nltk_tokens_list)
            standarized_sents += nltk_tokens_list

            o2n_map.append(list(range(n, n+n_splits)))
            n += n_splits

        else:
            standarized_sents.append(sents)
            o2n_map.append([n])
            n += 1
    embs = build_bert_emb(standarized_sents, tokenizer, model, args.device)

    combined_embs = list()
    for o2n in o2n_map:
        if len(o2n) == 1:
            combined_embs.append(embs[o2n[0]])
        else:
            cat_emb = torch.cat([embs[o2n[0]], embs[o2n[1]][1:], embs[o2n[2]][1:]], dim=0)
            combined_embs.append(cat_emb)

    for emb, sent in zip(combined_embs, token_list):
        assert len(emb) == len(sent) + 1

    print("[INFO] Embeddings Built!")
    return combined_embs


def main():
    args = parse_args()
    print(args)
    set_seed_everywhere(args.random_seed)

    args = build_config(args)

    string_list, token_list, anno_list = build_dataset(args)
    embs = build_embeddings(token_list, args)
    exp_sent_list = [["[CLS]"] + sent for sent in token_list]
    weak_lbs = [extract_sequence(
        s, a, sources=args.src_to_keep, label_indices=args.lbs2idx, ontonote_anno_scheme=False
    ) for s, a in zip(token_list, anno_list)]

    train_dataset = Dataset(
        text=exp_sent_list,
        embs=embs,
        obs=weak_lbs,
        lbs=None
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    args.d_emb = embs[0].size(-1)
    _, args.n_src, args.n_obs = weak_lbs[0].size()
    args.n_hidden = args.n_obs

    # inject prior knowledge about transition and emission
    state_prior = torch.zeros(args.n_hidden, device=args.device) + 1e-2
    state_prior[0] += 1 - state_prior.sum()

    intg_obs = list(map(np.array, weak_lbs))
    trans_mat = torch.tensor(initialise_transmat(
        observations=intg_obs, label_set=args.bio_lbs)[0], dtype=torch.float)
    emiss_mat = torch.tensor(initialise_emissions(
        observations=intg_obs, label_set=args.bio_lbs,
        sources=args.src_to_keep, src_priors=args.src_priors
    )[0], dtype=torch.float)

    if not args.model_dir:
        save_model = False
        model_file = None
    else:
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        model_file = os.path.join(args.model_dir, 'model.chkpt')
        save_model = True

    # ----- initialize model -----
    hmm_model = NeuralHMM(args=args, state_prior=state_prior, trans_matrix=trans_mat, emiss_matrix=emiss_mat)
    hmm_model.to(device=args.device)

    # ----- initialize optimizer -----
    hmm_params = [
        hmm_model.unnormalized_emiss,
        hmm_model.unnormalized_trans,
        hmm_model.state_priors
    ]
    optimizer = torch.optim.Adam(
        [{'params': hmm_model.nn_module.parameters(), 'lr': args.nn_lr},
         {'params': hmm_params}],
        lr=args.hmm_lr,
        weight_decay=1e-5
    )

    pretrain_optimizer = torch.optim.Adam(
        hmm_model.nn_module.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    # ----- initialize training process -----
    trainer = Trainer(hmm_model, args)

    # ----- pre-train neural module -----
    if args.pretrain_epoch > 0:
        print("[INFO] pre-training neural module")
        for epoch_i in range(args.pretrain_epoch):
            train_loss = trainer.pre_train(train_loader, pretrain_optimizer, trans_mat, emiss_mat)
            print(f"[INFO] Epoch: {epoch_i}, Loss: {train_loss}")

    # ----- start training process -----
    for epoch_i in range(args.epoch):
        print("========= Epoch %d of %d =========" % (epoch_i + 1, args.epoch))

        train_loss = trainer.train(train_loader, optimizer)

        print("========= Results: epoch %d of %d =========" % (epoch_i + 1, args.epoch))
        print("[INFO] train loss: %.4f" % train_loss)

    # ----- save model -----

    if save_model:
        model_state_dict = hmm_model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'settings': args
        }
        torch.save(checkpoint, model_file)
        print("[INFO] Checkpoint Saved!\n")

    scored_spans, _ = annotate_data(
        model=hmm_model, text=exp_sent_list, embs=embs,
        obs=weak_lbs, lbs=None, args=args
    )
    txt_span_list = list()
    for string, token, span in zip(string_list, token_list, scored_spans):
        txt_spans = token_to_txt_span(token, string, span)
        txt_span_list.append(txt_spans)
    with open('results.json', 'w') as f:
        str_span_list = list()
        for ss in txt_span_list:
            str_dict = dict()
            for key, v in ss.items():
                str_dict[str(key)] = (v[0][0], str(v[0][1]))
            str_span_list.append(str_dict)
        json.dump(str_span_list, f)


if __name__ == '__main__':
    main()
