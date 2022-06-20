import argparse
import torch
import os
import json

from TestCollection.NHMMTest import neural_hmm_test
from TestCollection.MajorityVoting import majority_voting
from TestCollection.SrcPerformance import source_performance
from TestCollection.IIDTest import iid_test
from TestCollection.HMMTest import hmm_test
from Core.Constants import *


def parse_args():
    """
    Wrapper function for parsing arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='indicates which dataset to use.'
    )
    parser.add_argument(
        '--test_task', type=str, default=r'multi-obs',
        help='indicates which file to test.'
    )
    parser.add_argument(
        '--data_dir', type=str, default='',
        help='training data location.'
    )
    parser.add_argument(
        '--train_name', type=str, default='',
        help='training data name.'
    )
    parser.add_argument(
        '--dev_name', type=str, default='',
        help='development data name.'
    )
    parser.add_argument(
        '--test_name', type=str, default='',
        help='test data name.'
    )
    parser.add_argument(
        '--model_dir', type=str, default='models',
        help='where to save checkpoints.'
    )
    parser.add_argument(
        '--figure_dir', type=str, default='plots',
        help='where to save checkpoints.'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs',
        help='where to save log files.'
    )
    parser.add_argument(
        '--trans_nn_weight', type=float, default=1, help='the weight of neural part in the transition matrix'
    )
    parser.add_argument(
        '--emiss_nn_weight', type=float, default=1, help='the weight of neural part in the emission matrix'
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
        '--obs_normalization', type=bool, default=True, help='not normalizing observations'
    )
    parser.add_argument(
        '--save_scores', action='store_true',
        help='store the denoised labels as scores into a seperate file'
    )
    parser.add_argument(
        '--converse_first', action='store_true', help='converse the labels to CoNLL 2003 domain first'
    )
    parser.add_argument(
        '--test_all', action='store_true', help='use the entire dataset as test set'
    )
    parser.add_argument(
        '--pin_memory', action='store_true', help='whether pin your cuda memory during training'
    )
    parser.add_argument(
        '--debugging_mode', action='store_true', help='output intermediate messages'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help='how many processes to use when loading data'
    )
    parser.add_argument(
        '--ontonote_anno_scheme', action='store_true', help='whether use ontonote annotation scheme'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print(args)

    return args


def set_dataset_specific_args(args):
    if not args.data_dir:
        args.data_dir = os.path.join('../data', args.dataset)
    if not args.train_name:
        args.train_name = args.dataset + '-linked-train.pt'
    if not args.dev_name:
        args.dev_name = args.dataset + '-linked-dev.pt'
    if not args.test_name:
        args.test_name = args.dataset + '-linked-test.pt'

    args.train_emb = args.train_name.replace('linked', 'emb')
    args.dev_emb = args.dev_name.replace('linked', 'emb')
    args.test_emb = args.test_name.replace('linked', 'emb')

    if args.dataset == 'Laptop':
        args.lbs = LAPTOP_LABELS
        args.bio_lbs = LAPTOP_BIO
        args.lbs2idx = LAPTOP_INDICES
        args.src = LAPTOP_SOURCE_NAMES
        args.src_to_keep = LAPTOP_SOURCES_TO_KEEP
        args.src_priors = LAPTOP_SOURCE_PRIORS
    elif args.dataset == 'NCBI':
        args.lbs = NCBI_LABELS
        args.bio_lbs = NCBI_BIO
        args.lbs2idx = NCBI_INDICES
        args.src = NCBI_SOURCE_NAMES
        args.src_to_keep = NCBI_SOURCES_TO_KEEP
        args.src_priors = NCBI_SOURCE_PRIORS
    elif args.dataset == 'BC5CDR':
        args.lbs = BC5CDR_LABELS
        args.bio_lbs = BC5CDR_BIO
        args.lbs2idx = BC5CDR_INDICES
        args.src = BC5CDR_SOURCE_NAMES
        args.src_to_keep = BC5CDR_SOURCES_TO_KEEP
        args.src_priors = BC5CDR_SOURCE_PRIORS
    elif args.dataset == 'Co03':
        args.lbs = CoNLL_LABELS
        args.bio_lbs = OntoNotes_BIO if not args.converse_first else CoNLL_BIO
        args.lbs2idx = OntoNotes_INDICES if not args.converse_first else CoNLL_INDICES
        args.src = CoNLL_SOURCE_NAMES
        args.src_to_keep = CoNLL_SOURCE_TO_KEEP
        args.src_priors = CoNLL_SOURCE_PRIORS if not args.converse_first else CONLL_SRC_PRIORS
        args.mappings = CoNLL_MAPPINGS
        args.prior_name = 'CoNLL03-init-stat-all.pt'
    else:
        json_name = os.path.join(args.data_dir, f'{args.dataset}-metadata.json')
        with open(json_name, 'r') as f:
            data = json.load(f)
        args.lbs = data['labels']
        if 'source-labels' not in data.keys():
            args.bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in args.lbs for bi in "BI"]
        else:
            args.bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in data['source-labels'] for bi in "BI"]
        args.lbs2idx = {label: i for i, label in enumerate(args.bio_lbs)}
        args.src = data['sources']
        args.src_to_keep = data['sources-to-keep'] if 'sources-to-keep' in data.keys() else data['sources']
        args.src_priors = data['priors'] if 'priors' in data.keys() else \
            {src: {lb: (0.8, 0.8) for lb in args.lbs} for src in args.src_to_keep}
        args.mappings = data['mapping'] if 'mapping' in data.keys() else None


def test(args):
    if args.test_task == 'nhmm':
        neural_hmm_test(args)
    elif args.test_task == 'source':
        source_performance(args)
    elif args.test_task == 'majority':
        majority_voting(args)
    elif args.test_task == 'iid':
        iid_test(args)
    elif args.test_task == 'hmm':
        hmm_test(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    arguments = parse_args()
    set_dataset_specific_args(arguments)
    test(arguments)
