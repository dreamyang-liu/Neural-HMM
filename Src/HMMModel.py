import pprint
import hmmlearn
import hmmlearn.hmm
import numpy as np

from Core.Data import label_to_span
from Core.Util import get_results, anno_space_map
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
        self.args = args
        self.informative_priors = informative_priors

        # initialize base class
        hmmlearn.hmm._BaseHMM.__init__(self, n_components=self.n_hidden, verbose=True, n_iter=self.epoch)

        self._set_state_dict(priors)

    def train(self, train_anno, dev_anno, dev_lbs, dev_sents):
        """Train the HMM annotator based on the docbin file"""

        pp = pprint.PrettyPrinter(indent=4)
        micro_results = list()
        log_results = dict()

        # Make sure the initialization is valid
        self._check()

        self.monitor_._reset()
        best_f1 = 0
        best_state_dict = self._get_state_dict()
        for i in range(self.epoch):
            print("[INFO] Starting iteration", (i + 1))
            curr_logprob = self.train_epoch(train_anno)

            self.monitor_.report(curr_logprob)

            test_results = self.test_epoch(dev_anno, dev_lbs, dev_sents)
            print("[INFO] validation results:")
            pp.pprint(test_results['micro'])

            if test_results['micro']['entity_f1'] > best_f1:
                best_f1 = test_results['micro']['entity_f1']
                best_state_dict = self._get_state_dict()

            micro_results.append(test_results)
            for k, v in test_results['micro'].items():
                if k not in log_results:
                    log_results[k] = list()
                log_results[k].append(v)

            if self.monitor_.converged:
                break
        self._set_state_dict(best_state_dict)

        return micro_results, log_results

    def train_epoch(self, weak_annotations):
        stats = self._initialize_sufficient_statistics()
        curr_logprob = 0

        n_insts = 0
        for annotations in tqdm(weak_annotations):
            # Actually this could be moved outside the loop (NO, unless we get all x in advance)
            framelogprob = self._compute_log_likelihood(annotations)
            if framelogprob.max(axis=1).min() < -100000:
                print("problem found!")
                return framelogprob
            # forward-backward training part
            logprob, fwdlattice = self._do_forward_pass(framelogprob)
            curr_logprob += logprob
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

    def test_epoch(self, weak_annotations, true_labels, sents):
        predictions = list()
        for annotation in weak_annotations:
            predictions.append(self.label(annotation)[0])
        pred_spans = list(map(label_to_span, predictions))
        if hasattr(self.args, 'mappings'):
            if self.args.mappings is not None:
                pred_spans = [anno_space_map(ps, self.args.mappings, self.args.lbs) for ps in pred_spans]
        true_spans = true_labels

        results = get_results(pred_spans, true_spans, sents, all_labels=self.args.lbs)

        return results

    def annotate(self, annotations):
        score_list = list()
        scored_spans_list = list()
        for annotation in annotations:
            labels, scores = self.label(annotation)
            indices = [self.args.lbs2idx[lb] for lb in labels]
            spans = label_to_span(labels)

            ps = [p[s] for p, s in zip(scores, indices)]

            scored_spans = dict()
            for k, v in spans.items():
                score = np.mean(ps[k[0]:k[1]])
                scored_spans[k] = [(v, score)]
            scored_spans_list.append(scored_spans)

            score_list.append(scores.astype(np.float32))

        return scored_spans_list, score_list

    def label(self, annotations):
        """Makes a list of predicted labels (using Viterbi) for each token, along with
        the associated probability according to the HMM model."""

        if not hasattr(self, "emission_probs"):
            raise RuntimeError("Model is not trained yet")

        framelogprob = self._compute_log_likelihood(annotations)
        logprob, predicted = self._do_viterbi_pass(framelogprob)
        # self.check_outputs(predicted)

        labels = [self.args.bio_lbs[x] for x in predicted]

        predicted_proba = np.exp(framelogprob)
        predicted_proba = predicted_proba / predicted_proba.sum(axis=1)[:, np.newaxis]

        return labels, predicted_proba

    def _set_state_dict(self, priors):
        self.startprob_prior = priors[1]
        self.startprob_ = priors[0]
        self.transmat_prior = priors[3]
        self.transmat_ = priors[2]
        self.emission_priors = priors[5]
        self.emission_probs = priors[4]
        return None

    def _get_state_dict(self):
        return [self.startprob_, self.startprob_prior,
                self.transmat_, self.transmat_prior,
                self.emission_probs, self.emission_priors]

    def _compute_log_likelihood(self, x):

        logsum = np.zeros((len(x), self.n_obs))
        for source_index in range(self.n_src):
            probs = np.dot(x[:, source_index, :], self.emission_probs[source_index, :, :].T)
            logsum += np.ma.log(probs).filled(-np.inf)

        # We also add a constraint that the probability of a state is zero is no labelling functions observes it
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

    def check_outputs(self, predictions):
        """Checks whether the output is consistent"""
        prev_bio_label = "O"
        for i in range(len(predictions)):
            bio_label = self.args.bio_lbs[predictions[i]]
            if prev_bio_label[0] == "O" and bio_label[0] == "I":
                print("inconsistent start of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
            elif prev_bio_label[0] in {"B", "I"}:
                if bio_label[0] == "I" and bio_label[2:] != prev_bio_label[2:]:
                    print("inconsistent continuation of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
            prev_bio_label = bio_label
