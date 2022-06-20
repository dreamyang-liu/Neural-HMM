import os
import torch
import pandas as pd
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, args, lr):
        self.model = model
        self.args = args
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.train_df = pd.DataFrame(columns=["loss", "lr"])
        self.valid_df = pd.DataFrame(columns=["loss", "lr"])

    def load_checkpoint(self):
        if os.path.isfile(os.path.join(self.args.model_dir, "model_state.pth")):
            try:
                self.model.load_state_dict(torch.load(
                    os.path.join(self.args.model_dir, "model_state.pth"), map_location=self.args.device
                ))
            except:
                print("Could not load previous model; starting from scratch")
        else:
            print("No previous model; starting from scratch")

    def save_checkpoint(self):
        try:
            torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, "model_state.pth"))
        except:
            print("[ERROR] Fail to save model")

    def train(self, dataset):
        train_acc = 0
        train_loss = 0
        num_samples = 0
        self.model.train()
        print_interval = 1000
        for idx, batch in enumerate(tqdm(dataset.loader)):
            x, lengths = batch
            batch_size = len(x)
            num_samples += batch_size
            log_probs = self.model(x, lengths)
            loss = -log_probs.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().data.numpy().item() * batch_size
            if idx % print_interval == 0:
                print(loss.item())
                for _ in range(5):
                    sampled_x, sampled_z = self.model.sample()
                    print("".join([self.args.obs_set[s] for s in sampled_x]))
                    print(sampled_z)
        train_loss /= num_samples
        train_acc /= num_samples
        return train_loss

    def test(self, dataset, print_interval=20):
        test_acc = 0
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
        test_acc /= num_samples
        self.scheduler.step(test_loss)  # if the validation loss hasn't decreased, lower the learning rate
        return test_loss
