import numpy as np

from matplotlib import pyplot as plt
from Core.Constants import CoNLL_LABELS


def plot_tran_emis(path, trans_, emiss_):
    plt.figure(figsize=(8, 6), dpi=200)
    plt.subplot(1, 2, 1)
    plt.imshow(trans_)
    plt.title('Transition matrix')
    plt.subplot(1, 2, 2)
    plt.imshow(emiss_)
    plt.title('Emission matrix')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return None


def plot_train_results(path: str, x: list):
    plt.figure()
    for xx in x:
        plt.plot(list(range(len(xx))), xx)
    plt.savefig(path, figsize=(8, 6), dpi=200, bbox_inches='tight')
    return None


def plot_detailed_results(path: str, result_list: list):
    entity_results = dict()
    for entity in CoNLL_LABELS + ['micro']:
        entity_results[entity] = {
            'token_f1': [],
            "token_precision": [],
            'token_recall': []
        }
    for result in result_list:
        for entity, metrics in entity_results.items():
            for metric in metrics:
                entity_results[entity][metric].append(result[entity][metric])

    fig = plt.figure(figsize=(8, 12), dpi=200)
    ax_micro = fig.add_subplot(3, 1, 3)
    for metric, value in entity_results['micro'].items():
        ax_micro.plot(np.arange(len(value)) + 1, value, label=metric)
    ax_micro.legend()
    ax_micro.set_title('micro')
    plt.grid(True)
    for i, entity in enumerate(CoNLL_LABELS):
        ax_entity = fig.add_subplot(3, 2, i+1)
        for metric, value in entity_results[entity].items():
            ax_entity.plot(np.arange(len(value)) + 1, value, label=metric)
        ax_entity.legend()
        ax_entity.set_title(entity)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')


def plot_dict(path: str, x: dict):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    for k, vals in x.items():
        ax.plot(list(range(len(vals))), vals, label=k)
    ax.legend()
    fig.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    return None


def plot_bar(path: str, results: dict):
    srcs = CoNLL_LABELS + ['micro']
    f1_list = list()
    precision_list = list()
    recall_list = list()
    for entity in srcs:
        f1_list.append(results[entity]['token_f1'])
        precision_list.append(results[entity]['token_precision'])
        recall_list.append(results[entity]['token_recall'])

    x = np.arange(len(srcs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    rect1 = ax.bar(x - width*1.05, precision_list, width, label='precision')
    rect2 = ax.bar(x, recall_list, width, label='recall')
    rect3 = ax.bar(x + width*1.05, f1_list, width, label='f1')

    ax.set_ylim([0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(srcs)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rect1)
    autolabel(rect2)
    autolabel(rect3)

    fig.tight_layout()
    plt.savefig(path, bbox_inches='tight')
