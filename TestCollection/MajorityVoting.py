import os
import torch
import pprint
import pandas as pd
import numpy as np

from Core.Util import set_seed_everywhere, get_results, one_hot
from Core.Data import span_to_label


def majority_voting(args):
    set_seed_everywhere(args.random_seed)
    if args.debugging_mode:
        torch.autograd.set_detect_anomaly(True)
    pp = pprint.PrettyPrinter(indent=4)

    img_dir = os.path.join(args.figure_dir, f"{args.dataset}.{args.test_task}")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)

    # ----- load data -----
    train_data = torch.load(os.path.join(args.data_dir, args.train_name))
    dev_data = torch.load(os.path.join(args.data_dir, args.dev_name))
    test_data = torch.load(os.path.join(args.data_dir, args.test_name))

    # ----- construct dataset -----
    sents = train_data['sentences'] + dev_data['sentences'] + test_data['sentences']
    annotations = train_data['annotations'] + dev_data['annotations'] + test_data['annotations']
    labels = train_data['labels'] + dev_data['labels'] + test_data['labels']
    len_train = len(train_data['sentences'])
    len_dev = len(dev_data['sentences'])
    len_test = len(test_data['sentences'])

    weak_spans = dict()
    for source in args.src_to_keep:
        src_anno_list = list()
        for annotation in annotations:
            anno_dict = dict()
            for k, v in annotation[source].items():
                if not v:
                    continue
                elif isinstance(v, str):
                    pass
                elif isinstance(v, tuple) or isinstance(v, list):
                    v = v[0][0]
                if hasattr(args, 'mappings'):
                    if args.mappings is not None:
                        pred_lb = v
                        norm_lb = args.mappings.get(pred_lb, pred_lb)
                        if norm_lb in args.lbs:
                            anno_dict[k] = norm_lb
                else:
                    anno_dict[k] = v
            src_anno_list.append(anno_dict)
        weak_spans[source] = src_anno_list

    major_spans = list()
    ensembled_spans = list()
    for i in range(len(list(weak_spans.values())[0])):
        temp_list = list()
        for v in weak_spans.values():
            temp_list += [v[i]]
        ensembled_spans.append(temp_list)
    for spans_set in ensembled_spans:
        temp_dict = dict()
        for spans in spans_set:
            for k, v in spans.items():
                if k not in temp_dict.keys():
                    temp_dict[k] = list()
                temp_dict[k].append(v)
        for k in temp_dict:
            temp_dict[k] = max(set(temp_dict[k]), key=temp_dict[k].count)
        major_spans.append(temp_dict)

    src_results = get_results(major_spans[-len_test:], labels[-len_test:], sents[-len_test:], all_labels=args.lbs)
    results = src_results['micro']

    pp.pprint(src_results)

    if args.debugging_mode:
        df = pd.DataFrame(data=[results])
        df.to_csv(os.path.join(args.log_dir, f"log.{args.dataset}.majority-voting.csv"))

    if args.save_scores:
        print("========= Saving Reliabilities =========")
        labels = [span_to_label(s, t) for s, t in zip(sents, major_spans)]
        indices = [[args.lbs2idx[lb] for lb in lbs] for lbs in labels]
        scores = [one_hot(np.array(x), len(args.lbs2idx)).numpy() for x in indices]

        pred_train_scores = scores[:len_train]
        pred_dev_scores = scores[len_train: len_train + len_dev]
        pred_test_scores = scores[-len_test:]

        scored_pred_span = list()
        for span_dict in major_spans:
            scored_span = dict()
            for k, v in span_dict.items():
                scored_span[k] = [(v, 1.0)]
            scored_pred_span.append(scored_span)

        pred_train_spans = scored_pred_span[:len_train]
        pred_dev_spans = scored_pred_span[len_train: len_train + len_dev]
        pred_test_spans = scored_pred_span[-len_test:]

        score_name = f"{'.'.join(args.train_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save((pred_train_spans, pred_train_scores), os.path.join(args.data_dir, score_name))

        score_name = f"{'.'.join(args.dev_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save((pred_dev_spans, pred_dev_scores), os.path.join(args.data_dir, score_name))

        score_name = f"{'.'.join(args.test_name.split('.')[:-1])}.{args.test_task}.scores"
        torch.save((pred_test_spans, pred_test_scores), os.path.join(args.data_dir, score_name))
        print("[INFO] Done")
