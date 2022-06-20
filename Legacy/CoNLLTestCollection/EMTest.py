import os
import torch
import numpy as np
import pandas as pd

from Legacy.CoNLL03 import HMMEM
from Core.Data import extract_sequence
from Core.Constants import SOURCE_NAMES_TO_KEEP, CoNLL_SOURCE_NAMES
from Core.Util import set_seed_everywhere
from Core.Display import plot_detailed_results


def em_test(args):
    set_seed_everywhere(args.random_seed)
    if args.debugging_mode:
        torch.autograd.set_detect_anomaly(True)

    # ----- load data -----
    train_data = torch.load(os.path.join(args.data_dir, args.data_name))
    dev_data = torch.load(os.path.join(args.data_dir, args.dev_name))

    # ----- construct dataset -----
    train_sents = train_data['sentences']
    train_annotations = train_data['annotations']

    train_labels = list()
    for doc_lbs in train_data['labels']:
        for lbs in doc_lbs:
            train_labels.append(lbs)
    train_weak_labels = [np.array(extract_sequence(s, a, SOURCE_NAMES_TO_KEEP), dtype=np.float)
                         for s, a in zip(train_sents, train_annotations)]
    
    dev_sents = dev_data['sentences']
    dev_annotations = dev_data['annotations']

    dev_labels = list()
    for doc_lbs in dev_data['labels']:
        for lbs in doc_lbs:
            dev_labels.append(lbs)
    dev_weak_labels = [np.array(extract_sequence(s, a, SOURCE_NAMES_TO_KEEP), dtype=np.float)
                         for s, a in zip(dev_sents, dev_annotations)]

    _, args.n_src, args.n_obs = train_weak_labels[0].shape
    args.n_hidden = args.n_obs

    # inject prior knowledge about transition and emission
    state_prior = torch.zeros(args.n_hidden, device=args.device) + 1e-2
    state_prior[0] += 1 - state_prior.sum()

    priors = torch.load(os.path.join(args.data_dir, args.prior_name))
    src_to_keep = [CoNLL_SOURCE_NAMES.index(idx) for idx in SOURCE_NAMES_TO_KEEP]
    priors['emission_matrix'] = priors['emission_matrix'][src_to_keep, :, :]
    priors['emission_strength'] = priors['emission_strength'][src_to_keep, :, :]

    # about saving checkpoint
    args_str = f"{args.test_task}"
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_file = os.path.join(args.model_dir, 'model.best.chkpt')
    img_dir = os.path.join(args.figure_dir, args_str)
    if args.debugging_mode and not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if args.debugging_mode and not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # ----- initialize model -----
    hmm_model = HMMEM(args=args, priors=priors)

    # ----- train and test model -----
    result_list, log_results = hmm_model.train(train_weak_labels, dev_weak_labels, dev_labels, dev_sents)
    if args.debugging_mode:
        plot_detailed_results(
            os.path.join(img_dir, 'accu.png'),
            result_list
        )
        df = pd.DataFrame(data=log_results)
        df.to_csv(os.path.join(args.log_dir, "log.{}.{}.csv".format(args.test_task, args_str)))
    return None
