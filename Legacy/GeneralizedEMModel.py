import torch
import torch.nn as nn
from Core.Util import log_matmul, log_maxmul, validate_prob, logsumexp
from typing import Optional


# TODO: This class only supports 1 as batch size
# TODO: Batched algorithm will be implemented in the future
class NeuralHMM(nn.Module):
    """
    Neural Hidden Markov Model.
    (For now, discrete obs_set only.)
    - forward(): computes the log probability of an observation sequence.
    - viterbi(): computes the most likely state sequence.
    - sample(): draws a sample from p(obs).
    """

    def __init__(self, args, state_prior=None, trans_matrix=None, emiss_matrix=None):
        super(NeuralHMM, self).__init__()

        self.args = args
        self.d_emb = args.d_emb  # embedding dimension
        self.n_obs = args.n_obs  # number of possible obs_set
        self.n_hidden = args.n_hidden  # number of states

        self._initialize_model(state_prior, trans_matrix, emiss_matrix)

    def _initialize_model(self,
                          state_prior: torch.Tensor,
                          trans_matrix: torch.Tensor,
                          emiss_matrix: torch.Tensor):

        if state_prior is None:
            priors = torch.zeros(self.n_hidden, device=self.args.device) + 1e-3
            priors[0] = 1
            self.state_priors = nn.Parameter(torch.log(priors))
        else:
            # TODO: beware of the -inf case
            state_prior.to(self.args.device)
            priors = validate_prob(state_prior, dim=0)
            self.state_priors = nn.Parameter(torch.log(priors))

        if trans_matrix is None:
            self.unnormalized_trans = nn.Parameter(torch.randn(self.n_hidden, self.n_hidden, device=self.args.device))
        else:
            trans_matrix.to(self.args.device)
            trans_matrix = validate_prob(trans_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_trans = nn.Parameter(torch.log(trans_matrix))

        if emiss_matrix is None:
            self.unnormalized_emiss = nn.Parameter(torch.randn(self.n_hidden, self.n_obs, device=self.args.device))
        else:
            emiss_matrix.to(self.args.device)
            emiss_matrix = validate_prob(emiss_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_emiss = nn.Parameter(torch.log(emiss_matrix))

        print("model initialized!")

        return None

    def _initialize_states(self,
                           length: int,
                           temperature: Optional[int] = 1.0):
        # normalize and put the probabilities into the log domain
        self.log_state_priors = torch.log_softmax(self.state_priors / temperature, dim=-1)
        self.log_trans = torch.log_softmax(self.unnormalized_trans / temperature, dim=-1)
        self.log_emiss = torch.log_softmax(self.unnormalized_emiss / temperature, dim=-1)

        self.log_alpha = torch.zeros([length, self.n_hidden], device=self.args.device)
        self.log_beta = torch.zeros([length, self.n_hidden], device=self.args.device)
        # Gamma can be readily computed and need no initialization
        self.log_gamma = None
        # only values in 1:max_seq_length are valid. The first state is a dummy
        self.log_xi = torch.zeros([length, self.n_hidden, self.n_hidden], device=self.args.device)
        return None

    def _forward_step(self, obs, t):
        # initial alpha state
        if t == 0:
            log_alpha_t = self.log_state_priors + log_matmul(torch.log(obs[t]), self.log_emiss.T).squeeze()
        # do the forward step
        else:
            log_alpha_t = log_matmul(torch.log(obs[t]), self.log_emiss.T).squeeze() + \
                          log_matmul(self.log_alpha[t - 1, :], self.log_trans).squeeze()

        # normalize the result
        normalized_log_alpha_t = log_alpha_t - log_alpha_t.logsumexp(dim=-1, keepdim=True)
        return normalized_log_alpha_t

    def _backward_step(self, obs, t):
        # do the backward step
        # beta is not a distribution, so we do not need to normalize it
        log_beta_t = log_matmul(
            self.log_trans, log_matmul(torch.log(obs[t]), self.log_emiss.T).T + self.log_beta[t + 1, :].unsqueeze(-1)
        ).squeeze()
        return log_beta_t

    def _forward_backward(self, obs, length):
        # calculate log alpha
        for t in range(0, length):
            self.log_alpha[t, :] = self._forward_step(obs, t)

        # calculate log beta
        # The last beta state beta[-1, :] = log1 = 0, so need to assign the value again
        for t in range(length-2, -1, -1):
            self.log_beta[t, :] = self._backward_step(obs, t)
        return None

    def _compute_xi(self, obs, t):
        temp_1 = log_matmul(torch.log(obs[t]), self.log_emiss.T) + self.log_beta[t, :].unsqueeze(0)
        temp_2 = log_matmul(self.log_alpha[t - 1, :].unsqueeze(-1), temp_1)
        log_xi_t = self.log_trans + temp_2
        return log_xi_t

    def _expected_complete_log_likelihood(self, obs, length):
        # calculate expected sufficient statistics: gamma_t(j) = P(z_t = j|x_{1:T})
        self.log_gamma = self.log_alpha + self.log_beta
        # normalize as gamma is a distribution
        log_gamma = self.log_gamma - self.log_gamma.logsumexp(dim=-1, keepdim=True)
        # calculate expected sufficient statistics: psi_t(i, j) = P(z_{t-1}=i, z_t=j|x_{1:T})
        for t in range(1, length):
            self.log_xi[t, :, :] = self._compute_xi(obs, t)
        # Normalization, t = 1:T (does not contain 0); use stable log_sum_exponential
        stabled_norm_term = logsumexp(self.log_xi[1:, :, :].view(self.log_xi.size(0)-1, -1), dim=1)\
            .view(self.log_xi.size(0)-1, 1, 1)
        log_xi = self.log_xi[1:, :, :] - stabled_norm_term

        # print("gamma:", log_gamma)
        # print("observation:", obs)
        # print("beta:", self.log_beta)
        # print("sum of xi:", stabled_norm_term.squeeze())
        # print("transition:", self.log_trans)
        # print("emission:", self.log_emiss)
        # print("xi:", log_xi)

        # calculate the expected complete data log likelihood
        log_prior = torch.sum(torch.exp(log_gamma[0, :]) * self.log_state_priors)
        log_tran = torch.sum(torch.exp(log_xi) * self.log_trans.unsqueeze(0))
        # log_tran = torch.logsumexp((self.log_xi[1:, :, :] + self.log_trans.unsqueeze(0)), dim=[0, 1, 2])
        log_emis = torch.sum(torch.exp(log_gamma) * log_matmul(torch.log(obs), self.log_emiss.T))
        log_likelihood = log_prior + log_tran + log_emis

        return log_likelihood

    def forward(self, obs):
        """
        obs : IntTensor of shape (batch size, max_length, n_src)
        lengths : IntTensor of shape (batch size)

        Compute log p(obs) for each example in the batch.
        lengths = max_seq_length of each example
        """
        # the row of obs should be one-hot or at least sum to 1
        assert (obs.sum(dim=-1) == 1).all()

        length = len(obs)
        # Initialize alpha and beta
        self._initialize_states(length=length)
        self._forward_backward(obs, length)
        log_likelihood = self._expected_complete_log_likelihood(obs, length)
        return log_likelihood

    def viterbi(self, obs):
        """
        obs : IntTensor of shape (batch size, max_len, n_src)
        lengths : IntTensor of shape (batch size)

        Find argmax_z log p(z|obs) for each (obs) in the batch.
        """
        length = len(obs)

        # initialize states
        self._initialize_states(length)
        # maximum probabilities
        log_delta = torch.zeros([length, self.n_hidden], device=self.args.device)
        # most likely previous state on the most probable path to z_t = j. a[0] is undefined.
        a = torch.zeros([length, self.n_hidden], dtype=torch.long, device=self.args.device)

        # the initial delta state
        log_delta[0, :] = self.log_state_priors + log_matmul(torch.log(obs[0]), self.log_emiss.T).squeeze()
        for t in range(1, length):
            # udpate delta and a. It does not matter where we put the emission probabilities
            max_log_prob, argmax_val = log_maxmul(
                log_delta[t-1, :],
                self.log_trans + log_matmul(torch.log(obs[t]), self.log_emiss.T)
            )
            log_delta[t, :] = max_log_prob.squeeze()
            a[t, :] = argmax_val.squeeze()

        # The terminal state
        max_log_prob, z_t_star = log_delta[-1, :].max(dim=0)

        # Trace back
        z_star = [z_t_star.item()]
        for t in range(length-2, -1, -1):
            z_t = a[t+1, z_star[0]].item()
            z_star.insert(0, z_t)

        return z_star, max_log_prob
