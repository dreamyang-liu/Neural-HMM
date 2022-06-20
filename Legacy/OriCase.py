import os
import torch
from Legacy.SimpleData import Dataset, collate_fn
from tqdm.auto import tqdm


class HMM(torch.nn.Module):
    """
    Hidden Markov Model with discrete observations.
    """

    def __init__(self, M, N):
        super(HMM, self).__init__()
        self.M = M  # number of possible observations
        self.N = N  # number of states

        # A
        self.transition_model = TransitionModel(self.N)

        # b(x_t)
        self.emission_model = EmissionModel(self.N, self.M)

        # pi
        self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.N))

        # use the GPU
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda: self.cuda()

    def forward(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)

        Compute log p(x) for each example in the batch.
        T = max_seq_length of each example
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]
        T_max = x.shape[1]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        log_alpha = torch.zeros(batch_size, T_max, self.N)
        if self.is_cuda: log_alpha = log_alpha.cuda()

        log_alpha[:, 0, :] = self.emission_model(x[:, 0]) + log_state_priors
        for t in range(1, T_max):
            log_alpha[:, t, :] = self.emission_model(x[:, t]) + self.transition_model(log_alpha[:, t - 1, :])

        # Select the sum for the final timestep (each x may have different max_seq_length).
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = torch.gather(log_sums, 1, T.view(-1, 1) - 1)
        return log_probs

    def sample(self, T=10):
        state_priors = torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
        # This might be wrong. I think A should also be normalized along dim 1?
        # Oh I see. So the transition matrix the author used is actually the transportation of the conventional one
        transition_matrix = torch.nn.functional.softmax(self.transition_model.unnormalized_transition_matrix, dim=0)
        emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=1)

        # sample initial state
        z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
        z = []
        x = []
        z.append(z_t)
        for t in range(0, T):
            # sample emission
            x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
            x.append(x_t)

            # sample transition
            z_t = torch.distributions.categorical.Categorical(transition_matrix[:, z_t]).sample().item()
            if t < T - 1: z.append(z_t)

        return x, z

    def viterbi(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)
        Find argmax_z log p(x|z) for each (x) in the batch.
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]
        T_max = x.shape[1]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        log_delta = torch.zeros(batch_size, T_max, self.N).float()
        psi = torch.zeros(batch_size, T_max, self.N).long()
        if self.is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()

        log_delta[:, 0, :] = self.emission_model(x[:, 0]) + log_state_priors
        for t in range(1, T_max):
            max_val, argmax_val = self.transition_model.maxmul(log_delta[:, t - 1, :])
            log_delta[:, t, :] = self.emission_model(x[:, t]) + max_val
            psi[:, t, :] = argmax_val

        # Get the log probability of the best path
        log_max = log_delta.max(dim=2)[0]
        best_path_scores = torch.gather(log_max, 1, T.view(-1, 1) - 1)

        # This next part is a bit tricky to parallelize across the batch,
        # so we will do it separately for each example.
        z_star = []
        for i in range(0, batch_size):
            z_star_i = [log_delta[i, T[i] - 1, :].max(dim=0)[1].item()]
            for t in range(T[i] - 1, 0, -1):
                z_t = psi[i, t, z_star_i[0]].item()
                z_star_i.insert(0, z_t)

            z_star.append(z_star_i)

        return z_star, best_path_scores  # return both the best path and its log probability


class TransitionModel(torch.nn.Module):
    def __init__(self, N):
        super(TransitionModel, self).__init__()
        self.N = N
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N, N))

    def forward(self, log_alpha):
        """
        log_alpha : Tensor of shape (batch size, N)
        Multiply previous timestep's alphas by transition matrix (in log domain)
        """
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

        # Matrix multiplication in the log domain
        out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0, 1)).transpose(0, 1)
        return out

    def maxmul(self, log_alpha):
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

        out1, out2 = maxmul(log_transition_matrix, log_alpha.transpose(0, 1))
        return out1.transpose(0, 1), out2.transpose(0, 1)


class EmissionModel(torch.nn.Module):
    def __init__(self, N, M):
        super(EmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N, M))

    def forward(self, x_t):
        log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
        out = log_emission_matrix[:, x_t].transpose(0, 1)
        return out


def log_domain_matmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.stack([log_A] * p, dim=2)
    log_B_expanded = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A_expanded + log_B_expanded
    out = torch.logsumexp(elementwise_sum, dim=1)

    return out


def maxmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Similar to the log domain matrix multiplication,
    this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.stack([log_A] * p, dim=2)
    log_B_expanded = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A_expanded + log_B_expanded
    out1, out2 = torch.max(elementwise_sum, dim=1)

    return out1, out2


class Trainer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

    def train(self, data_loader):
        train_loss = 0
        num_samples = 0
        self.model.train()
        for idx, batch in enumerate(tqdm(data_loader)):
            x, T = batch
            batch_size = len(x)
            num_samples += batch_size
            log_probs = self.model(x, T)
            loss = -log_probs.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().data.numpy().item() * batch_size
        train_loss /= num_samples
        return train_loss


def test(model, data_loader, idx2label):

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            obs_batch, seq_lens = batch
            # label_indices, confs = self.model.viterbi(emb_batch, obs_batch, seq_lens)
            label_indices, confs = model.inference(obs_batch, seq_lens)
            labels = [[idx2label[li] for li in lbs] for lbs in label_indices]
            true_obs = [[idx2label[li.item()] for li in lbs] for lbs in obs_batch]
            for n in range(len(obs_batch)):
                print(labels[n])
                print(true_obs[n])
                print()
            if i > 1:
                break
    return confs


def main():

    data_dir = r'data/dataset_ner/'
    data_name = 'dev.txt'

    with open(os.path.join(data_dir, data_name)) as f:
        lines = f.readlines()

    all_sentence = list()
    sentence = list()
    all_labels = list()
    labels = list()
    for l in lines:
        try:
            token, _, _, ner_label = l.strip().split()
            sentence.append(token)
            labels.append(ner_label)
        except ValueError:
            all_sentence.append(sentence)
            all_labels.append(labels)
            sentence = list()
            labels = list()

    for sentence, labels in zip(all_sentence, all_labels):
        assert len(sentence) == len(labels)

    label2idx = dict()
    for labels in all_labels:
        for l in labels:
            if l not in label2idx.keys():
                label2idx[l] = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}

    lb_indices = [[0] + [label2idx[lb] for lb in lbs] for lbs in all_labels]

    data_set = Dataset(obs=lb_indices)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        num_workers=0,
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=False,
        drop_last=False
    )

    model = HMM(N=len(idx2label), M=len(idx2label))

    # Train the model
    num_epochs = 10
    trainer = Trainer(model, lr=0.005)

    for epoch in range(num_epochs):
        print("========= Epoch %d of %d =========" % (epoch + 1, num_epochs))
        train_loss = trainer.train(data_loader)

        print("========= Results: epoch %d of %d =========" % (epoch + 1, num_epochs))
        print("train loss: %.2f" % (train_loss))

    test(model, data_loader, idx2label)


if __name__ == '__main__':
    main()
