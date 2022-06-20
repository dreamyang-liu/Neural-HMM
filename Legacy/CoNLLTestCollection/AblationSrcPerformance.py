import os
import torch
import pprint
import pandas as pd

from Core.Data import label_to_span
from Core.Constants import SOURCE_NAMES_TO_KEEP, CoNLL_MAPPINGS, CONLL_TO_RETAIN
from Core.Util import set_seed_everywhere, get_results
from Core.Display import plot_bar


def source_performance(args):
    set_seed_everywhere(args.random_seed)
    if args.debugging_mode:
        torch.autograd.set_detect_anomaly(True)
    pp = pprint.PrettyPrinter(indent=4)

    img_dir = os.path.join(args.figure_dir, f"{args.test_task}")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)

    # ----- load data -----
    data = torch.load(os.path.join(args.data_dir, args.dev_name))

    # ----- construct dataset -----
    sents = data['sentences']
    annotations = data['annotations']

    labels = list()
    for doc_lbs in data['labels']:
        for lbs in doc_lbs:
            labels.append(lbs)
    # _ = [sent.insert(0, "[CLS]") for sent in sents]
    # _ = [lbs.insert(0, "O") for lbs in labels]

    true_spans = [label_to_span(lbs) for lbs in labels]
    weak_spans = dict()
    for source in SOURCE_NAMES_TO_KEEP:
        src_anno_list = list()
        for annotation in annotations:
            anno_dict = dict()
            for k, v in annotation[source].items():
                value = v[0]
                norm_span = CoNLL_MAPPINGS.get(value[0], value[0])
                if norm_span in CONLL_TO_RETAIN:
                    anno_dict[k] = norm_span
            src_anno_list.append(anno_dict)
        weak_spans[source] = src_anno_list

    results = dict()
    for k, spans in weak_spans.items():
        src_results = get_results(spans, true_spans, sents)
        for key, value in src_results['micro'].items():
            if key not in results:
                results[key] = list()
            results[key].append(value)
        print(k)
        pp.pprint(src_results)
        plot_bar(os.path.join(img_dir, f'{k}.png'), src_results)

    df = pd.DataFrame(data=results, index=weak_spans.keys())
    df.to_csv(os.path.join(args.log_dir, "ablation-source-performance.csv"))
