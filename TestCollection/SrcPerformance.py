import os
import torch
import pprint
import pandas as pd

from Core.Util import set_seed_everywhere, get_results


def source_performance(args):
    set_seed_everywhere(args.random_seed)
    if args.debugging_mode:
        torch.autograd.set_detect_anomaly(True)
    pp = pprint.PrettyPrinter(indent=4)

    img_dir = os.path.join(args.figure_dir, f"{args.test_task}")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    # ----- load data -----
    test_data = torch.load(os.path.join(args.data_dir, args.test_name))

    # ----- construct dataset -----
    sents = test_data['sentences']
    annotations = test_data['annotations']
    annotator_names = list(annotations[0].keys())

    true_spans = test_data['labels']

    weak_spans = dict()
    for source in annotator_names:
        src_anno_list = list()
        for annotation in annotations:
            anno_dict = dict()
            for k, v in annotation[source].items():
                if isinstance(v, str):
                    pass
                elif isinstance(v, tuple) or isinstance(v, list):
                    v = v[0][0]
                if args.dataset == 'Co03':
                    pred_lb = v
                    norm_lb = args.mappings.get(pred_lb, pred_lb)
                    if norm_lb in args.lbs:
                        anno_dict[k] = norm_lb
                else:
                    anno_dict[k] = v
            src_anno_list.append(anno_dict)
        weak_spans[source] = src_anno_list

    results = dict()
    for k, spans in weak_spans.items():
        src_results = get_results(spans, true_spans, sents, all_labels=args.lbs)
        for key, value in src_results['micro'].items():
            if key not in results:
                results[key] = list()
            results[key].append(value)
        print(k)
        pp.pprint(src_results)

    df = pd.DataFrame(data=results, index=weak_spans.keys())
    df.to_csv(os.path.join(args.log_dir, f"log.{args.dataset}.source-performance.csv"))
