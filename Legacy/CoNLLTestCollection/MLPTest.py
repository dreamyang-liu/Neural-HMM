import os
import torch
import pprint
import pandas as pd

from torch.utils.data import DataLoader
from Legacy.CoNLL03.MLPModel import NeuralHMM
from Legacy.CoNLL03.MultiObsTrain import Trainer
from Legacy.CoNLL03 import Dataset, collate_fn
from Core.Data import extract_sequence
from Core.Constants import SOURCE_NAMES_TO_KEEP, CoNLL_SOURCE_NAMES
from Core.Util import set_seed_everywhere
from Core.Display import plot_tran_emis, plot_detailed_results


def mlp_test(args):
    set_seed_everywhere(args.random_seed)
    if args.debugging_mode:
        torch.autograd.set_detect_anomaly(True)

    pp = pprint.PrettyPrinter(indent=4)

    # ----- load train_data -----
    train_data = torch.load(os.path.join(args.data_dir, args.data_name))
    dev_data = torch.load(os.path.join(args.data_dir, args.dev_name))

    # ----- construct dataset -----
    train_sents = train_data['sentences']
    train_embs = train_data['embeddings']
    train_annotations = train_data['annotations']

    train_labels = list()
    for doc_lbs in train_data['labels']:
        for lbs in doc_lbs:
            train_labels.append(lbs)
    train_weak_labels = [extract_sequence(s, a, SOURCE_NAMES_TO_KEEP) for s, a in zip(train_sents, train_annotations)]
    _ = [sent.insert(0, "[CLS]") for sent in train_sents]
    _ = [lbs.insert(0, "O") for lbs in train_labels]

    train_dataset = Dataset(text=train_sents, embs=train_embs, obs=train_weak_labels, lbs=train_labels)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    # ----- construct development dataset -----
    dev_sents = dev_data['sentences']
    dev_embs = dev_data['embeddings']
    dev_annotations = dev_data['annotations']

    dev_labels = list()
    for doc_lbs in dev_data['labels']:
        for lbs in doc_lbs:
            dev_labels.append(lbs)
    dev_weak_labels = [extract_sequence(s, a, SOURCE_NAMES_TO_KEEP) for s, a in zip(dev_sents, dev_annotations)]
    _ = [sent.insert(0, "[CLS]") for sent in dev_sents]
    _ = [lbs.insert(0, "O") for lbs in dev_labels]

    dev_dataset = Dataset(text=dev_sents, embs=dev_embs, obs=dev_weak_labels, lbs=dev_labels)
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    args.d_emb = train_embs[0].size(-1)
    _, args.n_src, args.n_obs = train_weak_labels[0].size()
    args.n_hidden = args.n_obs

    # inject prior knowledge about transition and emission
    state_prior = torch.zeros(args.n_hidden, device=args.device) + 1e-2
    state_prior[0] += 1 - state_prior.sum()

    priors = torch.load(os.path.join(args.data_dir, args.prior_name))
    trans_mat = torch.tensor(priors['transition_matrix'], dtype=torch.float)
    emiss_mat_all = torch.tensor(priors['emission_matrix'], dtype=torch.float)
    emiss_mat = emiss_mat_all[[CoNLL_SOURCE_NAMES.index(idx) for idx in SOURCE_NAMES_TO_KEEP], :, :]

    # about saving checkpoint
    args_str = f"{args.test_task}.{args.trans_nn_weight}-{args.emiss_nn_weight}-{args.hmm_lr}-{args.nn_lr}"
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    model_file = os.path.join(args.model_dir, f'{args.test_task}.model.best.chkpt')
    img_dir = os.path.join(args.figure_dir, args_str)
    if args.debugging_mode and not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if args.debugging_mode and not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    if args.debugging_mode:
        plot_tran_emis(
            os.path.join(img_dir, f'initial_matrix.pdf'),
            trans_mat,
            emiss_mat.mean(dim=0)
        )

    # ----- initialize model -----
    hmm_model = NeuralHMM(args=args, state_prior=state_prior, trans_matrix=trans_mat, emiss_matrix=emiss_mat)
    hmm_model.to(device=args.device)

    # ----- initialize optimizer -----
    hmm_params = [
        hmm_model.unnormalized_emiss,
        hmm_model.unnormalized_trans,
        hmm_model.state_priors
    ]
    optimizer = torch.optim.Adam([
        {'params': hmm_model.nn_module.parameters(), 'lr': args.nn_lr},
        {'params': hmm_params}
    ],
        lr=args.hmm_lr,
        weight_decay=1e-5
    )

    pre_train_optimizer = torch.optim.Adam(
        hmm_model.nn_module.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    # ----- initialize training process -----
    trainer = Trainer(hmm_model, args)
    best_f1 = 0
    micro_results = list()
    log_results = dict()

    # ----- pre-train neural module -----
    if args.pre_train:
        print("[INFO] pre-training neural module")
        for epoch_i in range(args.pretrain_epoch):
            train_loss = trainer.pre_train(train_loader, pre_train_optimizer, trans_mat, emiss_mat)
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

        # ----- save model -----

        f1 = (results['micro']['token_f1'] + results['micro']['entity_f1']) / 2
        if f1 >= best_f1:
            model_state_dict = hmm_model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'optimizer': optimizer_state_dict,
                'settings': args,
                'epoch': epoch_i
            }
            torch.save(checkpoint, model_file)
            print("[INFO] Checkpoint Saved!\n")

            best_f1 = f1

        # ----- log history -----

        micro_results.append(results)
        for k, v in results['micro'].items():
            if k not in log_results:
                log_results[k] = list()
            log_results[k].append(v)

        if args.debugging_mode:
            plot_tran_emis(
                os.path.join(img_dir, 'matrix.{:02d}.png'.format(epoch_i)),
                torch.softmax(hmm_model.unnormalized_trans, dim=-1).detach().cpu().numpy(),
                torch.softmax(hmm_model.unnormalized_emiss.mean(dim=0), dim=-1).detach().cpu().numpy()
            )

    if args.debugging_mode:
        plot_detailed_results(
            os.path.join(img_dir, 'accu.png'),
            micro_results
        )
        df = pd.DataFrame(data=log_results)
        df.to_csv(os.path.join(args.log_dir, "log.{}.{}.csv".format(args.test_task, args_str)))
