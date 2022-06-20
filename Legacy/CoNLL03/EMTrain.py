import pprint
import hmmlearn
import hmmlearn.hmm
import numpy as np

from Core.Constants import OntoNotes_BIO
from Core.Data import label_to_span
from Core.Constants import CoNLL_MAPPINGS, CONLL_TO_RETAIN
from Core.Util import get_results
from numba import njit, prange
from tqdm.auto import tqdm


# noinspection PyProtectedMember,PyArgumentList,PyTypeChecker
class HMMEM(hmmlearn.hmm._BaseHMM):

    def __init__(self,
                 args,
                 priors,
                 informative_priors=True):

        self.n_hidden = args.n_hidden
        self.n_src = args.n_src
        self.n_obs = args.n_obs
        self.epoch = args.epoch
        self.informative_priors = informative_priors

        # initialize base class
        hmmlearn.hmm._BaseHMM.__init__(self, n_components=self.n_hidden, verbose=True, n_iter=self.epoch)

        self._set_priors(priors)

    def train(self, train_anno, dev_anno, dev_labels, dev_sents):
        """Train the HMM annotator based on the docbin file"""

        pp = pprint.PrettyPrinter(indent=4)
        micro_results = list()
        log_results = dict()

        # Make sure the initialization is valid
        self._check()

        self.monitor_._reset()
        for i in range(self.epoch):
            print("[INFO] Starting iteration", (i + 1))
            curr_logprob = self._train_epoch(train_anno)

            self.monitor_.report(curr_logprob)

            test_results = self._test_epoch(dev_anno, dev_labels, dev_sents)
            print("[INFO] test results:")
            pp.pprint(test_results['micro'])

            micro_results.append(test_results)
            for k, v in test_results['micro'].items():
                if k not in log_results:
                    log_results[k] = list()
                log_results[k].append(v)

            if self.monitor_.converged:
                break

        return micro_results, log_results

    def _train_epoch(self, weak_annotations):
        stats = self._initialize_sufficient_statistics()
        curr_logprob = 0

        n_insts = 0
        for annotations in tqdm(weak_annotations):
            # TODO: This function is re-implemented by the author
            # Actually this could be moved outside the loop (NO, unless we get all x in advance)
            framelogprob = self._compute_log_likelihood(annotations)
            if framelogprob.max(axis=1).min() < -100000:
                print("problem found!")
                return framelogprob
            # forward-backward training part
            # TODO: this is a hmmlearn function
            logprob, fwdlattice = self._do_forward_pass(framelogprob)
            curr_logprob += logprob
            # TODO: these are hmmlearn functions
            bwdlattice = self._do_backward_pass(framelogprob)
            posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
            self._accumulate_sufficient_statistics(
                stats, annotations, framelogprob, posteriors, fwdlattice,
                bwdlattice)
            n_insts += 1

        print("Finished E-step with %i documents" % n_insts)

        # XXX must be before convergence check, because otherwise
        #     there won't be any updates for the case ``n_iter=1``.
        self._do_mstep(stats)
        return curr_logprob

    def _test_epoch(self, weak_annotations, true_labels, sents):
        predictions = list()
        for annotation in weak_annotations:
            predictions.append(self.label(annotation)[0])
        pred_spans = list(map(label_to_span, predictions))

        pred_spans_norm = list()
        for spans in pred_spans:
            normalized_span = dict()
            for k in spans:
                norm_span = CoNLL_MAPPINGS.get(spans[k], spans[k])
                if norm_span in CONLL_TO_RETAIN:
                    normalized_span[k] = norm_span
            pred_spans_norm += [normalized_span]

        true_spans = list(map(label_to_span, true_labels))

        results = get_results(pred_spans_norm, true_spans, sents)

        return results

    def label(self, annotations):
        """Makes a list of predicted labels (using Viterbi) for each token, along with
        the associated probability according to the HMM model."""

        if not hasattr(self, "emission_probs"):
            raise RuntimeError("Model is not yet trained")

        framelogprob = self._compute_log_likelihood(annotations)
        logprob, predicted = self._do_viterbi_pass(framelogprob)
        self.check_outputs(predicted)

        labels = [OntoNotes_BIO[x] for x in predicted]

        predicted_proba = np.exp(framelogprob)
        predicted_proba = predicted_proba / predicted_proba.sum(axis=1)[:, np.newaxis]

        confidences = np.array([probs[x] for (probs, x) in zip(predicted_proba, predicted)])
        return labels, confidences

    def _set_priors(self, priors):
        self.startprob_prior = priors['state_prior_count']
        self.startprob_ = priors['state_prior']
        self.transmat_prior = priors['transition_count']
        self.transmat_ = priors['transition_matrix']
        self.emission_priors = priors['emission_strength']
        self.emission_probs = priors['emission_matrix']
        return None

    def _compute_log_likelihood(self, x):

        logsum = np.zeros((len(x), self.n_obs))
        for source_index in range(self.n_src):
            probs = np.dot(x[:, source_index, :], self.emission_probs[source_index, :, :].T)
            logsum += np.ma.log(probs).filled(-np.inf)

        # We also add a constraint that the probability of a state is zero is no labelling functions observes it
        # TODO: If no labelling functions observes it, set the emission prob to 0
        # TODO: What is the influence of this trick? It seems infeasible when back-propagation is adopted
        x_all_obs = x.sum(axis=1).astype(bool)
        logsum = np.where(x_all_obs, logsum, -np.inf)

        return logsum

    def _initialize_sufficient_statistics(self) -> dict:
        stats = super(HMMEM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros(self.emission_probs.shape)
        return stats

    def _accumulate_sufficient_statistics(self, stats, x, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(HMMEM, self)._accumulate_sufficient_statistics(
            stats, x, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            self.sum_posteriors(stats["obs"], x, posteriors)

    def _do_mstep(self, stats):
        super(HMMEM, self)._do_mstep(stats)
        if 'e' in self.params:
            emission_counts = self.emission_priors + stats['obs']
            emission_probs = emission_counts / (emission_counts + 1E-100).sum(axis=2)[:, :, np.newaxis]
            self.emission_probs = np.where(self.emission_probs > 0, emission_probs, 0)

    @staticmethod
    @njit(parallel=True)
    def sum_posteriors(stats, x, posteriors):
        for i in prange(x.shape[0]):
            for source_index in range(x.shape[1]):
                for j in range(x.shape[2]):
                    obs = x[i, source_index, j]
                    if obs > 0:
                        stats[source_index, :, j] += (obs * posteriors[i])

    @staticmethod
    def check_outputs(predictions):
        """Checks whether the output is consistent"""
        prev_bio_label = "O"
        for i in range(len(predictions)):
            bio_label = OntoNotes_BIO[predictions[i]]
            if prev_bio_label[0] == "O" and bio_label[0] == "I":
                print("inconsistent start of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
            elif prev_bio_label[0] in {"B", "I"}:
                if bio_label[0] not in {"I", "O"}:
                    print("inconsistent continuation of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
                if bio_label[0] == "I" and bio_label[2:] != prev_bio_label[2:]:
                    print("inconsistent continuation of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
            prev_bio_label = bio_label
