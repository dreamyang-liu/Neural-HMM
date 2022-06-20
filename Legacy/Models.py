import torch
import torch.nn as nn
import torch.nn.functional as F
from Core.Util import log_matmul, log_maxmul


class NeuralTransition(nn.Module):
    """
    Neural Transition Model
    """

    def __init__(self, d_emb, n_hidden):
        super(NeuralTransition, self).__init__()
        self.n_hidden = n_hidden  # number of states

        self.emb2h = nn.Linear(d_emb, n_hidden)  # h should be an n_hidden dimensional vector
        self.h2score = nn.Linear(n_hidden, 1)  # score should be a scalar
        self.h2u = nn.Conv1d(1, n_hidden, 1)  # u should be an n_hidden by n_hidden matrix

        self.unnormalized_tran = nn.Parameter(torch.randn(n_hidden, n_hidden))
        self._init_tran()

    def _init_tran(self):
        print("Transition matrix initialized!")

    def forward(self, emb, log_alpha, use_max):
        """
        log_alpha : Tensor of shape (batch size, n_hidden_states)

        Multiply previous timestep's alphas by transition matrix (in log domain)
        """
        # Each col needs to add up to 1 (in probability domain)
        h = F.gelu(self.emb2h(emb))  # s_batch by n_hidden

        score = torch.sigmoid(self.h2score(h)).view([-1, 1, 1])  # s_batch

        # TODO: another normalization function might be more reasonable.
        u = torch.softmax(self.h2u(h.unsqueeze(1)), dim=-1)
        tran = torch.softmax(self.unnormalized_tran, dim=-1)

        compound_log_tran = torch.log(score * u + (1 - score) * tran.repeat([len(score), 1, 1]))

        # Matrix multiplication in the log domain
        if use_max:
            out1, out2 = log_maxmul(log_alpha.unsqueeze(1), compound_log_tran)
            return out1.squeeze(1), out2.squeeze(1)
        else:
            out = log_matmul(log_alpha.unsqueeze(1), compound_log_tran).squeeze(1)
            return out


class NeuralEmission(nn.Module):
    """
    - forward(): computes the log probability of an observation.
    - sample(): given a state, sample an observation for that state.
    """

    def __init__(self, n_hidden, n_src, n_obs):
        super(NeuralEmission, self).__init__()
        self.n_hidden = n_hidden  # number of states
        self.n_obs = n_obs  # number of possible observations

        self.unnormalized_emis = nn.Parameter(torch.randn(n_src, n_hidden, n_obs))
        self._init_emis()

    def _init_emis(self):
        print("Emission matrix initialized")

    def forward(self, o_t):
        """
        o_t : LongTensor of shape (batch_size by n_src)

        Get observation probabilities
        """
        # Each row needs to add up to 1 (in probability domain)
        emission_matrices = F.log_softmax(self.unnormalized_emis, dim=-1)
        # operate each annotation source
        emission_list = list()
        for i in range(emission_matrices.size(0)):
            emission_list.append(emission_matrices[i, :, o_t.T[i, :]])
        # s_batch by n_hidden
        out = torch.stack(emission_list).permute([2, 0, 1]).mean(dim=1)

        return out


class NeuralHMM(nn.Module):
    """
    Neural Hidden Markov Model.
    (For now, discrete obs_set only.)
    - forward(): computes the log probability of an observation sequence.
    - viterbi(): computes the most likely state sequence.
    - sample(): draws a sample from p(obs).
    """

    def __init__(self, args):
        super(NeuralHMM, self).__init__()

        self.args = args
        self.d_emb = args.d_emb  # embedding dimension
        self.n_obs = args.n_obs  # number of possible obs_set
        self.n_src = args.n_src  # number of weak labeling sources
        self.n_hidden = args.n_hidden  # number of states

        self.transition_model = NeuralTransition(self.d_emb, self.n_hidden)
        self.emission_model = NeuralEmission(self.n_hidden, self.n_src, self.n_obs)

        self.log_state_priors = None
        self._init_hidden_states()

        self.is_cuda = args.cuda
        if self.is_cuda:
            self.cuda()

    def _init_hidden_states(self):
        priors = torch.zeros(self.n_hidden, device=self.args.device) + 0.1
        priors[0] = 1
        self.log_state_priors = F.log_softmax(priors, dim=0)
        print("hidden states initialized!")

    def forward(self, embs, obs, lengths):
        """
        obs : IntTensor of shape (batch size, max_length, n_src)
        lengths : IntTensor of shape (batch size)

        Compute log p(obs) for each example in the batch.
        lengths = max_seq_length of each example
        """
        if self.is_cuda:
            embs = embs.cuda()
            obs = obs.cuda()
            lengths = lengths.cuda()

        s_batch, max_length, _ = obs.size()
        log_alpha = torch.zeros([s_batch, max_length, self.n_hidden], device=self.args.device)

        log_alpha[:, 0, :] = self.emission_model(obs[:, 0, :]) + self.log_state_priors
        # print(log_alpha[:, 0, :])
        for t in range(1, max_length):
            log_alpha[:, t, :] = self.emission_model(obs[:, t, :]) + self.transition_model(
                emb=embs[:, t, :], log_alpha=log_alpha[:, t - 1, :], use_max=False
            )
            # print(log_alpha[:, t, :])

        log_sums = log_alpha.logsumexp(dim=2)

        # Select the sum for the final timestep (each obs has different max_seq_length).
        log_probs = torch.gather(log_sums, 1, lengths.view(-1, 1) - 1)
        return log_probs

    def viterbi(self, embs, obs, lengths):
        """
        obs : IntTensor of shape (batch size, max_len, n_src)
        lengths : IntTensor of shape (batch size)

        Find argmax_z log p(z|obs) for each (obs) in the batch.
        """
        if self.is_cuda:
            embs = embs.cuda()
            obs = obs.cuda()
            lengths = lengths.cuda()

        s_batch, max_len, _ = obs.size()
        log_delta = torch.zeros([s_batch, max_len, self.n_hidden], device=self.args.device)
        psi = torch.zeros([s_batch, max_len, self.n_hidden], dtype=torch.long, device=self.args.device)

        log_delta[:, 0, :] = self.emission_model(obs[:, 0, :]) + self.log_state_priors
        for t in range(1, max_len):
            max_val, argmax_val = self.transition_model(
                emb=embs[:, t, :], log_alpha=log_delta[:, t - 1, :], use_max=True
            )
            log_delta[:, t, :] = self.emission_model(obs[:, t, :]) + max_val
            psi[:, t, :] = argmax_val

        # Get the probability of the best data_path
        log_max = log_delta.max(dim=2)[0]
        best_path_scores = torch.gather(log_max, 1, lengths.view(-1, 1) - 1)

        # This next part is a bit tricky to parallelize across the batch,
        # so we will do it separately for each example.
        z_star = []
        for i in range(0, s_batch):
            z_star_i = [log_delta[i, lengths[i] - 1, :].max(dim=0)[1].item()]
            for t in range(lengths[i] - 1, 0, -1):
                z_t = psi[i, t, z_star_i[0]].item()
                z_star_i.insert(0, z_t)

            z_star.append(z_star_i)

        return z_star, best_path_scores
