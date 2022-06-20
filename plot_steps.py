import os
import torch

from torch.utils.data import DataLoader
from Legacy.CoNLL03 import NeuralHMM
from Legacy.CoNLL03 import Dataset, collate_fn
from Core.Data import extract_sequence
from Core.Constants import SOURCE_NAMES_TO_KEEP
from matplotlib import pyplot as plt


def plot_tran_emis(path, trans_, emiss_, word, label):
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(trans_)
    ax1.set_title(f'Transition: "{word}"-"{label}"')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(emiss_)
    ax2.set_title(f'Emission: "{word}"-"{label}"')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return None


model_dir = r'models/multi-obs.model.best.chkpt'
chkpt = torch.load(model_dir)

args = chkpt['settings']
hmm_model = NeuralHMM(args)
hmm_model.load_state_dict(chkpt['model'])
hmm_model.to(args.device)

# ----- load data -----
data = torch.load(os.path.join(args.data_dir, args.data_name))

# ----- construct dataset -----
sents = data['sentences']
embs = data['embeddings']
annotations = data['annotations']

labels = list()
for doc_lbs in data['labels']:
    for lbs in doc_lbs:
        labels.append(lbs)
weak_labels = [extract_sequence(s, a, SOURCE_NAMES_TO_KEEP) for s, a in zip(sents, annotations)]
_ = [sent.insert(0, "[CLS]") for sent in sents]
_ = [lbs.insert(0, "O") for lbs in labels]

data_set = Dataset(text=sents, embs=embs, obs=weak_labels, lbs=labels)
data_loader = torch.utils.data.DataLoader(
    dataset=data_set,
    num_workers=0,
    batch_size=1,
    collate_fn=collate_fn,
    shuffle=True,
    pin_memory=False,
    drop_last=False
)

img_dir = r'plots/steps'
if args.debugging_mode and not os.path.isdir(img_dir):
    os.mkdir(img_dir)

for i, batch in enumerate(data_loader):

    if i > 0:
        break

    hmm_model.eval()
    with torch.no_grad():
        emb_batch, obs_batch, seq_lens = map(lambda x: x.to(args.device), batch[:3])
        sent = batch[-2]
        true_lbs = batch[-1]

        nn_trans, nn_emiss = hmm_model.nn_module(embs=emb_batch)
        print(sent)
        print(true_lbs)
        print(nn_trans.shape, nn_emiss.shape)
        nn_trans = nn_trans.squeeze().cpu().numpy()
        nn_emiss = nn_emiss.squeeze().mean(dim=-3).cpu().numpy()

        for n, (token, lbs, tran, emis) in enumerate(zip(
                sent[0][1:], true_lbs[0][1:], nn_trans[1:], nn_emiss[1:]
        )):
            print('!!')
            plot_tran_emis(os.path.join(img_dir, f'{n + 1}.png'), tran, emis, token, lbs)
