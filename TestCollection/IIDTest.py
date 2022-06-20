import os
import torch
import numpy as np
import pprint
import pandas as pd

from torch.utils.data import DataLoader
from collections import Counter
from Src.IIDModel import NeuralIID
from Src.IIDTrain import Trainer
from Src.Data import Dataset, collate_fn, annotate_data
from Core.Data import extract_sequence, initialise_emissions
from Core.Util import set_seed_everywhere
from Core.Display import plot_dict


def iid_test(args):
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
    train_embs = torch.load(os.path.join(args.data_dir, args.train_emb))
    train_annotations = train_data['annotations']
    train_lbs = train_data['labels']
    train_weak_lbs = [extract_sequence(
        s, a, sources=args.src_to_keep, label_indices=args.lbs2idx, ontonote_anno_scheme=ontonote_anno_scheme
    ) for s, a in zip(train_sents, train_annotations)]
    exp_train_sents = [["[CLS]"] + sent for sent in train_sents]

    dev_sents = dev_data['sentences']
    dev_embs = torch.load(os.path.join(args.data_dir, args.dev_emb))
    dev_annotations = dev_data['annotations']
    dev_lbs = dev_data['labels']
    dev_weak_lbs = [extract_sequence(
        s, a, sources=args.src_to_keep, label_indices=args.lbs2idx, ontonote_anno_scheme=ontonote_anno_scheme
    ) for s, a in zip(dev_sents, dev_annotations)]
    exp_dev_sents = [["[CLS]"] + sent for sent in dev_sents]

    test_sents = test_data['sentences']
    test_embs = torch.load(os.path.join(args.data_dir, args.test_emb))
    test_annotations = test_data['annotations']
    test_lbs = test_data['labels']
    test_weak_lbs = [extract_sequence(
        s, a, sources=args.src_to_keep, label_indices=args.lbs2idx, ontonote_anno_scheme=ontonote_anno_scheme
    ) for s, a in zip(test_sents, test_annotations)]
    exp_test_sents = [["[CLS]"] + sent for sent in test_sents]

    train_dataset = Dataset(
        text=exp_train_sents,
        embs=train_embs,
        obs=train_weak_lbs,
        lbs=train_lbs
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

    dev_dataset = Dataset(
        text=exp_dev_sents,
        embs=dev_embs,
        obs=dev_weak_lbs,
        lbs=dev_lbs
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    if args.test_all:
        test_dataset = Dataset(
            text=exp_train_sents + exp_dev_sents + exp_test_sents,
            embs=train_embs + dev_embs + test_embs,
            obs=train_weak_lbs + dev_weak_lbs + test_weak_lbs,
            lbs=train_lbs + dev_lbs + test_lbs
        )
    else:
        test_dataset = Dataset(text=exp_test_sents, embs=test_embs, obs=test_weak_lbs, lbs=test_lbs)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    args.d_emb = train_embs[0].size(-1)
    _, args.n_src, args.n_obs = train_weak_lbs[0].size()
    args.n_hidden = args.n_obs

    # inject prior knowledge about transition and emission
    init_counts = np.zeros(args.n_obs)
    for weak_lb in train_weak_lbs:
        cnts = Counter(weak_lb.argmax(dim=-1).view(-1).tolist())
        for k, v in cnts.items():
            init_counts[k] += v
    init_counts += 1
    state_prior = np.random.dirichlet(init_counts + 1E-10)
    state_prior = torch.tensor(state_prior, dtype=torch.float)

    intg_obs = list(map(np.array, train_weak_lbs+dev_weak_lbs))
    emiss_mat = torch.tensor(initialise_emissions(
        observations=intg_obs, label_set=args.bio_lbs,
        sources=args.src_to_keep, src_priors=args.src_priors
    )[0], dtype=torch.float)

    # about saving checkpoint
    args_str = f"{args.dataset}.{args.test_task}." \
               f"{args.trans_nn_weight}-{args.emiss_nn_weight}-{args.hmm_lr}-{args.nn_lr}"
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_file = os.path.join(args.model_dir, f'{args.dataset}.{args.test_task}.model.chkpt')
    img_dir = os.path.join(args.figure_dir, args_str)
    if args.debugging_mode and not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if args.debugging_mode and not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # ----- initialize model -----
    iid_model = NeuralIID(args=args, emiss_matrix=emiss_mat)
    iid_model.to(device=args.device)

    # ----- initialize optimizer -----
    hmm_params = [
        iid_model.unnormalized_emiss
    ]
    optimizer = torch.optim.Adam(
        [{'params': iid_model.nn_module.parameters(), 'lr': args.nn_lr},
         {'params': hmm_params}],
        lr=args.hmm_lr,
        weight_decay=1e-5
    )

    pre_train_optimizer = torch.optim.Adam(
        iid_model.nn_module.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    # ----- initialize training process -----
    trainer = Trainer(iid_model, args)
    micro_results = list()
    log_results = dict()
    best_f1 = 0

    # ----- pre-train neural module -----
    if args.pretrain_epoch > 0:
        print("[INFO] pre-training neural module")
        for epoch_i in range(args.pretrain_epoch):
            train_loss = trainer.pre_train(train_loader, pre_train_optimizer, state_prior, emiss_mat)
            print(f"[INFO] Epoch: {epoch_i}, Loss: {train_loss}")

    # ----- start training process -----
    for epoch_i in range(args.epoch):
        print("========= Epoch %d of %d =========" % (epoch_i + 1, args.epoch))

        train_loss = trainer.train(train_loader, optimizer)
        results = trainer.test(dev_loader)

        print("========= Results: epoch %d of %d =========" % (epoch_i + 1, args.epoch))
        print("[INFO] train loss: %.4f" % train_loss)
        print("[INFO] test results:")
        pp.pprint(results['micro'])

        if results['micro']['entity_f1'] > best_f1:
            model_state_dict = iid_model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'optimizer': optimizer_state_dict,
                'settings': args,
                'epoch': epoch_i
            }
            torch.save(checkpoint, model_file)
            print("[INFO] Checkpoint Saved!\n")
            best_f1 = results['micro']['entity_f1']

        micro_results.append(results)

        for k, v in results['micro'].items():
            if k not in log_results:
                log_results[k] = list()
            log_results[k].append(v)

    print("========= Final Test Results =========")
    iid_model.load_state_dict(torch.load(model_file)['model'])
    results = trainer.test(test_loader)
    pp.pprint(results['micro'])
    micro_results.append(results)
    for k, v in results['micro'].items():
        if k not in log_results:
            log_results[k] = list()
        log_results[k].append(v)

    if args.debugging_mode:
        print("========= Saving Logs =========")
        plot_dict(os.path.join(img_dir, 'f1.png'), log_results)

        df = pd.DataFrame(data=log_results)
        df.to_csv(os.path.join(args.log_dir, "log.{}.csv".format(args_str)))

    if args.save_scores:
        print("========= Saving Reliabilities =========")
        train_scores = annotate_data(
            model=iid_model, text=exp_train_sents, embs=train_embs,
            obs=train_weak_lbs, lbs=train_lbs, args=args
        )
        score_name = f"{'.'.join(args.train_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save(train_scores, os.path.join(args.data_dir, score_name))

        dev_scores = annotate_data(
            model=iid_model, text=exp_dev_sents, embs=dev_embs,
            obs=dev_weak_lbs, lbs=dev_lbs, args=args
        )
        score_name = f"{'.'.join(args.dev_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save(dev_scores, os.path.join(args.data_dir, score_name))

        test_scores = annotate_data(
            model=iid_model, text=exp_test_sents, embs=test_embs,
            obs=test_weak_lbs, lbs=test_lbs, args=args
        )
        score_name = f"{'.'.join(args.test_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save(test_scores, os.path.join(args.data_dir, score_name))
        print("[INFO] Done")
