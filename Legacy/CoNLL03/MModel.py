import torch
import torch.nn as nn
from Core.Util import log_matmul, validate_prob
from Core.Constants import OntoNotes_BIO
from Core.Data import label_to_span
from typing import Optional


class NeuralModule(nn.Module):
    def __init__(self,
                 d_emb,
                 n_hidden):
        super(NeuralModule, self).__init__()

        self.n_hidden = n_hidden
        self.neural_transition = nn.Linear(d_emb, self.n_hidden * self.n_hidden)

        self._init_parameters()

    def forward(self,
                embs: torch.Tensor,
                temperature: Optional[int] = 1.0):
        batch_size, max_seq_length, _ = embs.size()
        trans_temp = self.neural_transition(embs).view(
            batch_size, max_seq_length, self.n_hidden, self.n_hidden
        )
        nn_trans = torch.softmax(trans_temp / temperature, dim=-1)

        return nn_trans

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.neural_transition.weight.data, gain=nn.init.calculate_gain('relu'))


class NeuralHMM(nn.Module):
    """
    Neural Hidden Markov Model.
    (For now, discrete obs_set only.)
    - forward(): computes the log probability of an observation sequence.
    - viterbi(): computes the most likely state sequence.
    """

    def __init__(self,
                 args,
                 state_prior=None,
                 trans_matrix=None):
        super(NeuralHMM, self).__init__()

        self.d_emb = args.d_emb  # embedding dimension
        self.n_hidden = args.n_hidden  # number of states

        self.trans_weight = args.trans_nn_weight
        self.emiss_weight = args.emiss_nn_weight

        self.device = args.device

        self.nn_module = NeuralModule(d_emb=self.d_emb, n_hidden=self.n_hidden)

        # initialize unnormalized state-prior, transition and emission matrices
        self._initialize_model(
            state_prior=state_prior, trans_matrix=trans_matrix
        )

    def _initialize_model(self,
                          state_prior: torch.Tensor,
                          trans_matrix: torch.Tensor,
                          ):

        if state_prior is None:
            priors = torch.zeros(self.n_hidden, device=self.device) + 1E-3
            priors[0] = 1
            self.state_priors = nn.Parameter(torch.log(priors))
        else:
            # TODO: beware of the -inf case
            state_prior.to(self.device)
            priors = validate_prob(state_prior, dim=0)
            self.state_priors = nn.Parameter(torch.log(priors))

        if trans_matrix is None:
            self.unnormalized_trans = nn.Parameter(torch.randn(self.n_hidden, self.n_hidden, device=self.device))
        else:
            trans_matrix.to(self.device)
            trans_matrix = validate_prob(trans_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_trans = nn.Parameter(torch.log(trans_matrix))

        print("[INFO] model initialized!")

        return None

    def _initialize_states(self,
                           embs: torch.Tensor,
                           temperature: Optional[int] = 1.0):
        # normalize and put the probabilities into the log domain
        self.log_state_priors = torch.log_softmax(self.state_priors / temperature, dim=-1)
        trans = torch.softmax(self.unnormalized_trans / temperature, dim=-1)

        # get neural transition and emission matrices
        # TODO: we can add layer-norm later to see what happens
        nn_trans = self.nn_module(embs)

        # TODO: the coefficients are subject to change
        self.log_trans = torch.log((1-self.trans_weight) * trans + self.trans_weight * nn_trans)
        return None

    def forward(self, emb):
        """
        obs : IntTensor of shape (batch size, max_length, n_src)
        lengths : IntTensor of shape (batch size)

        Compute log p(obs) for each example in the batch.
        lengths = max_seq_length of each example
        """
        # the row of obs should be one-hot or at least sum to 1
        # assert (obs.sum(dim=-1) == 1).all()

        batch_size, max_seq_length, _ = emb.size()

        # Initialize alpha, beta and xi
        self._initialize_states(embs=emb)
        log_hidden_states = torch.zeros([batch_size, max_seq_length, self.n_hidden], device=self.device)

        log_hidden_states[:, 0, :] = self.state_priors.unsqueeze(0).repeat(batch_size, 1)
        for t in range(1, max_seq_length):
            log_hidden_states[:, t, :] = log_matmul(
                log_hidden_states[:, t-1, :].unsqueeze(-2), self.log_trans[:, t, :, :]
            ).squeeze()

        return log_hidden_states
