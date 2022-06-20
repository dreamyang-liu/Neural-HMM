import os
import torch
import pprint
import numpy as np
import pandas as pd

from Src.HMMModel import HMMEM
from Core.Data import extract_sequence, initialise_startprob, initialise_transmat, initialise_emissions, \
    converse_ontonote_to_conll
from Core.Util import set_seed_everywhere


def hmm_test(args):
    set_seed_everywhere(args.random_seed)
    pp = pprint.PrettyPrinter(indent=4)

    # ----- load data -----
    train_data = torch.load(os.path.join(args.data_dir, args.train_name))
    dev_data = torch.load(os.path.join(args.data_dir, args.dev_name))
    test_data = torch.load(os.path.join(args.data_dir, args.test_name))

    # ----- construct dataset -----
    ontonote_anno_scheme = True if (args.dataset == 'Co03' and not args.converse_first) or \
                                   args.ontonote_anno_scheme else False

    train_sents = train_data['sentences']
    train_annotations = train_data['annotations']
    if args.converse_first:
        train_annotations = converse_ontonote_to_conll(args, train_annotations)
    train_lbs = train_data['labels']
    train_weak_lbs = [np.array(extract_sequence(
        s, a, sources=args.src_to_keep, label_indices=args.lbs2idx, ontonote_anno_scheme=ontonote_anno_scheme
    )) for s, a in zip(train_sents, train_annotations)]

    if args.obs_normalization:
        for obs in train_weak_lbs:
            lbs = obs.argmax(axis=-1)
            # at least one source observes an entity
            entity_idx = lbs.sum(axis=-1) != 0
            # the sources that do not observe any entity
            no_entity_idx = lbs == 0
            no_obs_src_idx = entity_idx[:, np.newaxis] * no_entity_idx
            subsitute_prob = np.zeros_like(obs[0, 0])
            subsitute_prob[0] = 0.01
            subsitute_prob[1:] = 0.99 / obs.shape[-1]
            obs[no_obs_src_idx] = subsitute_prob

    dev_sents = dev_data['sentences']
    dev_annotations = dev_data['annotations']
    if args.converse_first:
        dev_annotations = converse_ontonote_to_conll(args, dev_annotations)
    dev_lbs = dev_data['labels']
    dev_weak_lbs = [np.array(extract_sequence(
        s, a, sources=args.src_to_keep, label_indices=args.lbs2idx, ontonote_anno_scheme=ontonote_anno_scheme
    )) for s, a in zip(dev_sents, dev_annotations)]

    if args.obs_normalization:
        for obs in dev_weak_lbs:
            lbs = obs.argmax(axis=-1)
            # at least one source observes an entity
            entity_idx = lbs.sum(axis=-1) != 0
            # the sources that do not observe any entity
            no_entity_idx = lbs == 0
            no_obs_src_idx = entity_idx[:, np.newaxis] * no_entity_idx
            subsitute_prob = np.zeros_like(obs[0, 0])
            subsitute_prob[0] = 0.01
            subsitute_prob[1:] = 0.99 / obs.shape[-1]
            obs[no_obs_src_idx] = subsitute_prob

    test_sents = test_data['sentences']
    test_annotations = test_data['annotations']
    if args.converse_first:
        test_annotations = converse_ontonote_to_conll(args, test_annotations)
    test_lbs = test_data['labels']
    test_weak_lbs = [np.array(extract_sequence(
        s, a, sources=args.src_to_keep, label_indices=args.lbs2idx, ontonote_anno_scheme=ontonote_anno_scheme
    )) for s, a in zip(test_sents, test_annotations)]

    if args.obs_normalization:
        for obs in test_weak_lbs:
            lbs = obs.argmax(axis=-1)
            # at least one source observes an entity
            entity_idx = lbs.sum(axis=-1) != 0
            # the sources that do not observe any entity
            no_entity_idx = lbs == 0
            no_obs_src_idx = entity_idx[:, np.newaxis] * no_entity_idx
            subsitute_prob = np.zeros_like(obs[0, 0])
            subsitute_prob[0] = 0.01
            subsitute_prob[1:] = 0.99 / obs.shape[-1]
            obs[no_obs_src_idx] = subsitute_prob

    _, args.n_src, args.n_obs = train_weak_lbs[0].shape
    args.n_hidden = args.n_obs

    # inject prior knowledge about transition and emission

    state_prior = torch.zeros(args.n_hidden, device=args.device) + 1e-2
    state_prior[0] += 1 - state_prior.sum()

    intg_obs = list(map(np.array, train_weak_lbs + dev_weak_lbs + test_weak_lbs))
    startprob_, startprob_prior = initialise_startprob(observations=intg_obs, label_set=args.bio_lbs)
    transmat_, transmat_prior = initialise_transmat(observations=intg_obs, label_set=args.bio_lbs)
    emission_probs, emission_priors = initialise_emissions(
        observations=intg_obs, label_set=args.bio_lbs,
        sources=args.src_to_keep, src_priors=args.src_priors
    )
    priors = [startprob_, startprob_prior, transmat_, transmat_prior, emission_probs, emission_priors]

    # about saving checkpoint
    args_str = f"{args.dataset}.{args.test_task}"
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    img_dir = os.path.join(args.figure_dir, args_str)
    if args.debugging_mode and not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if args.debugging_mode and not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # ----- initialize model -----
    hmm_model = HMMEM(args=args, priors=priors)

    # ----- train and test model -----
    train_weak_labels = train_weak_lbs + dev_weak_lbs + test_weak_lbs
    if args.test_all:
        test_weak_labels = train_weak_lbs + dev_weak_lbs + test_weak_lbs
        test_labels = train_lbs + dev_lbs + test_lbs
        test_sentences = train_sents + dev_sents + test_sents
    else:
        test_weak_labels = test_weak_lbs
        test_labels = test_lbs
        test_sentences = test_sents

    result_list, log_results = hmm_model.train(
        train_weak_labels,
        dev_weak_lbs, dev_lbs, dev_sents
    )
    results = hmm_model.test_epoch(test_weak_labels, test_labels, test_sentences)
    print("[INFO] test results:")
    pp.pprint(results['micro'])

    for k, v in results['micro'].items():
        if k not in log_results:
            log_results[k] = list()
        log_results[k].append(v)

    if args.debugging_mode:
        print("========= Saving Logs =========")
        df = pd.DataFrame(data=log_results)
        df.to_csv(os.path.join(args.log_dir, "log.{}.csv".format(args_str)))

    if args.save_scores:
        print("========= Saving Reliabilities =========")
        pred_train_scores = hmm_model.annotate(train_weak_lbs)
        pred_dev_scores = hmm_model.annotate(dev_weak_lbs)
        pred_test_scores = hmm_model.annotate(test_weak_lbs)

        score_name = f"{'.'.join(args.train_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save(pred_train_scores, os.path.join(args.data_dir, score_name))

        score_name = f"{'.'.join(args.dev_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save(pred_dev_scores, os.path.join(args.data_dir, score_name))

        score_name = f"{'.'.join(args.test_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save(pred_test_scores, os.path.join(args.data_dir, score_name))
        print("[INFO] Done")
