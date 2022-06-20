import torch
import torch.nn as nn
import numpy as np
from Core.Util import log_matmul, validate_prob
from Core.Data import label_to_span
from typing import Optional


class NeuralModule(nn.Module):
    def __init__(self,
                 d_emb,
                 n_hidden,
                 n_src,
                 n_obs):
        super(NeuralModule, self).__init__()

        self.n_hidden = n_hidden
        self.n_src = n_src
        self.n_obs = n_obs
        self.neural_hidden = nn.Linear(d_emb, self.n_hidden)
        self.neural_emissions = nn.ModuleList([
            nn.Linear(d_emb, self.n_hidden * self.n_obs) for _ in range(self.n_src)
        ])

        self._init_parameters()

    def forward(self,
                embs: torch.Tensor,
                temperature: Optional[int] = 1.0):
        batch_size, max_seq_length, _ = embs.size()
        hidden_logits = self.neural_hidden(embs)
        nn_hidden_states = torch.softmax(hidden_logits / temperature, dim=-1)

        nn_emiss = torch.stack([torch.softmax(emiss(embs).view(
                batch_size, max_seq_length, self.n_hidden, self.n_obs
            ) / temperature, dim=-1) for emiss in self.neural_emissions]).permute(1, 2, 0, 3, 4)
        return nn_hidden_states, nn_emiss

    def _init_parameters(self):
        for emiss in self.neural_emissions:
            nn.init.xavier_uniform_(emiss.weight.data, gain=nn.init.calculate_gain('relu'))


class NeuralIID(nn.Module):

    def __init__(self,
                 args,
                 emiss_matrix=None):
        super(NeuralIID, self).__init__()

        self.d_emb = args.d_emb  # embedding dimension
        self.n_src = args.n_src
        self.n_obs = args.n_obs  # number of possible obs_set
        self.n_hidden = args.n_hidden  # number of states

        self.trans_weight = args.trans_nn_weight
        self.emiss_weight = args.emiss_nn_weight

        self.device = args.device

        self.nn_module = NeuralModule(d_emb=self.d_emb, n_hidden=self.n_hidden, n_src=self.n_src, n_obs=self.n_obs)

        # initialize unnormalized state-prior, transition and emission matrices
        self._initialize_model(
            emiss_matrix=emiss_matrix
        )

    def _initialize_model(self,
                          emiss_matrix: torch.Tensor
                          ):

        if emiss_matrix is None:
            # self.unnormalized_emiss = nn.Parameter(torch.randn(self.n_hidden, self.n_obs, device=self.device))
            self.unnormalized_emiss = nn.Parameter(torch.zeros(self.n_hidden, self.n_obs, device=self.device))
        else:
            emiss_matrix.to(self.device)
            emiss_matrix = validate_prob(emiss_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_emiss = nn.Parameter(torch.log(emiss_matrix))

        print("[INFO] model initialized!")

        return None

    def _initialize_states(self,
                           embs: torch.Tensor,
                           obs: torch.Tensor,
                           temperature: Optional[int] = 1.0,
                           normalize_observation=True):
        # normalize and put the probabilities into the log domain
        batch_size, max_seq_length, n_src, _ = obs.size()
        emiss = torch.softmax(self.unnormalized_emiss / temperature, dim=-1)

        # get neural transition and emission matrices
        # TODO: we can add layer-norm later to see what happens
        self.hidden_states, nn_emiss = self.nn_module(embs)

        # TODO: the coefficients are subject to change
        self.log_emiss = torch.log((1-self.emiss_weight) * emiss + self.emiss_weight * nn_emiss)

        # if at least one source observes an entity at a position, set the probabilities of other sources to
        # the mean value (so that they will not affect the prediction)
        # maybe we can also set them to 0?
        if normalize_observation:
            lbs = obs.argmax(dim=-1)
            # at least one source observes an entity
            entity_idx = lbs.sum(dim=-1) != 0
            # the sources that do not observe any entity
            no_entity_idx = lbs == 0
            no_obs_src_idx = entity_idx.unsqueeze(-1) * no_entity_idx
            subsitute_prob = torch.zeros_like(obs[0, 0, 0])
            subsitute_prob[0] = 0.01
            subsitute_prob[1:] = 0.99 / self.n_obs
            obs[no_obs_src_idx] = subsitute_prob

        self.log_emiss_probs = log_matmul(
            self.log_emiss, torch.log(obs).unsqueeze(-1)
        ).squeeze(-1).sum(dim=-2)

        return None

    def forward(self, emb, obs, seq_lengths, normalize_observation):
        """
        obs : IntTensor of shape (batch size, max_length, n_src)
        lengths : IntTensor of shape (batch size)

        Compute log p(obs) for each example in the batch.
        lengths = max_seq_length of each example
        """
        # the row of obs should be one-hot or at least sum to 1
        # assert (obs.sum(dim=-1) == 1).all()

        batch_size, max_seq_length, n_src, n_obs = obs.size()
        assert n_obs == self.n_obs
        assert n_src == self.n_src

        # Initialize alpha, beta and xi
        self._initialize_states(embs=emb, obs=obs, normalize_observation=normalize_observation)
        marginal = torch.logsumexp(torch.log(self.hidden_states) + self.log_emiss_probs, dim=-1)
        log_likelihood = torch.stack([inst[:length].sum() for inst, length in zip(marginal, seq_lengths)])
        log_likelihood = log_likelihood.mean()
        return log_likelihood, self.log_emiss

    def inference(self, emb, obs, seq_lengths, normalize_observation):
        """
        obs : IntTensor of shape (batch size, max_len, n_src)
        seq_lengths : IntTensor of shape (batch size)

        Find argmax_z log p(z|obs) for each (obs) in the batch.
        """

        # initialize states
        self._initialize_states(embs=emb, obs=obs, normalize_observation=normalize_observation)
        joint_prob = torch.exp(torch.log(self.hidden_states) + self.log_emiss_probs)
        joint_prob = joint_prob / joint_prob.sum(dim=-1, keepdim=True)
        _, z_stars = joint_prob.max(dim=-1)

        batch_indices = list()
        batch_probs = list()
        for z, probs, length in zip(z_stars, joint_prob, seq_lengths):
            batch_indices.append(z[:length].detach().cpu().numpy())
            batch_probs.append(probs[1:length].detach().cpu().numpy())

        return batch_indices, batch_probs

    def annotate(self, emb, obs, seq_lengths, label_set, normalize_observation):
        batch_label_indices, batch_probs = self.inference(emb, obs, seq_lengths, normalize_observation)
        batch_labels = [[label_set[lb_index] for lb_index in label_indices]
                        for label_indices in batch_label_indices]

        # For batch_spans, we are going to compare them with the true spans,
        # and the true spans is already shifted, so we do not need to shift predicted spans back=
        batch_spans = list()
        batch_scored_spans = list()
        for labels, probs, indices in zip(batch_labels, batch_probs, batch_label_indices):
            spans = label_to_span(labels)
            batch_spans.append(spans)

            ps = [p[s] for p, s in zip(probs, indices[1:])]
            scored_spans = dict()
            for k, v in spans.items():
                if k == (0, 1):
                    continue
                start = k[0] - 1 if k[0] > 0 else 0
                end = k[1] - 1
                score = np.mean(ps[start:end])
                scored_spans[(start, end)] = [(v, score)]
            batch_scored_spans.append(scored_spans)

        return batch_spans, (batch_scored_spans, batch_probs)
