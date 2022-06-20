import os
import torch
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def load_checkpoint(self):
        if os.path.isfile(os.path.join(self.args.model_dir, "HMM.chkpt")):
            try:
                chkpt = torch.load(
                    os.path.join(self.args.model_dir, "HMM.chkpt")
                )
                params = chkpt['model-param']
                self.model.load_state_dict(params)
            except:
                print("[ERROR] Fail to load previous model; starting from scratch")
        else:
            print("No previous model; starting from scratch")

    def save_checkpoint(self):
        try:
            arguments = self.args
            to_save = {
                'model-param': self.model.state_dict(),
                'args': arguments
            }
            torch.save(to_save, os.path.join(self.args.model_dir, "HMM.chkpt"))
        except:
            print("[ERROR] Fail to save model")

    def train(self, data_loader):

        train_loss = 0
        num_samples = 0

        self.model.train()

        for batch in tqdm(data_loader):
            emb_batch, obs_batch, seq_lens, _ = batch
            batch_size = len(emb_batch)
            num_samples += batch_size
            log_probs = self.model(embs=emb_batch, obs=obs_batch, lengths=seq_lens)
            # log_probs = self.model(obs=obs_batch, lengths=seq_lens)
            loss = -log_probs.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().data.numpy().item() * batch_size
        train_loss /= num_samples

        print(train_loss)

        return train_loss

    def validate(self, dataset):
        test_loss = 0
        num_samples = 0
        self.model.eval()
        print_interval = 1000
        for idx, batch in enumerate(dataset.loader):
            x, T = batch
            batch_size = len(x)
            num_samples += batch_size
            log_probs = self.model(x, T)
            loss = -log_probs.mean()
            test_loss += loss.cpu().data.numpy().item() * batch_size
            if idx % print_interval == 0:
                print(loss.item())
                sampled_x, sampled_z = self.model.sample()
                print("".join([self.args.obs_set[s] for s in sampled_x]))
                print(sampled_z)
        test_loss /= num_samples
        self.scheduler.step(test_loss)  # if the validation loss hasn't decreased, lower the learning rate
        return test_loss

    def test(self, data_loader, idx2label):

        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                emb_batch, obs_batch, seq_lens, tokens = batch
                label_indices, confs = self.model.inference(emb_batch, obs_batch, seq_lens)
                # label_indices, confs = self.model.viterbi(obs_batch, seq_lens)
                labels = [[idx2label[li] for li in lbs] for lbs in label_indices]
                for n in range(len(tokens)):
                    print(' '.join(tokens[n]))
                    print(labels[n])
                if i > 1:
                    break
        return confs

