import math
import psutil

import numpy as np
from scipy.stats import chi2, binom_test
from scipy.stats import mannwhitneyu as mwu_scipy
from sklearn.feature_extraction.text import TfidfTransformer


class StatisticalAnalysis:

    def __init__(self, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant, vocabulary):

        self.dtm_count_all = dtm_count_all
        self.dtm_count_plaintiff = dtm_count_plaintiff
        self.dtm_count_defendant = dtm_count_defendant
        self.vocabulary = vocabulary

        # create and store term-frequency normalized document-term-matrices.
        tf_transformer_sides = TfidfTransformer(use_idf=False)
        tf_transformer_sides.fit(dtm_count_all)
        self.dtm_tf_plaintiff = tf_transformer_sides.transform(self.dtm_count_plaintiff.copy())
        self.dtm_tf_defendant = tf_transformer_sides.transform(self.dtm_count_defendant.copy())
        True

    def frequency_ratio(self):
        """
        Calculates the frequency ratio for each term (how much more often does a term appear in the
        plaintiff than in the defendant corpus. To get the same value for the defendant corpus,
        calculate 1/frequency_ratio)

        Given a term t, the frequency ratio is calculated as follows
        frequency_ratio(t) = frequency_in_plaintiff_docs(t) / frequency_in_defendant_docs(t)

        The p-value calculation takes the difference in corpus sizes between plaintiff and
        defendant corpora into account

        >>> statistical_analysis = load_dummy_data()
        Loading six test documents
        >>> freq, freq_p = statistical_analysis.frequency_ratio()

        Within the dummy documents (see corpus.py, _load_test_corpus() function) most words appear
        in equal numbers in the plaintiff and defendant dummy documents
        Hence, the word "and" gets a Frequency Ratio 1 and a p-value of 1.
        >>> and_index = statistical_analysis.vocabulary.index('and')
        >>> 'And: Frequency Ratio: {}. p-value: {}'.format(freq[and_index], freq_p[and_index])
        'And: Frequency Ratio: 1.0. p-value: 1.0'

        The only terms that show divergence are 'plaintiff' and 'defendant'
        "Defendant" only appears in defendant documents and has a frequency ratio of 0
        >>> def_index = statistical_analysis.vocabulary.index('defendant')
        >>> 'Defendant: Frequency Ratio: {}. p-value: {}'.format(freq[def_index], freq_p[def_index])
        'Defendant: Frequency Ratio: 0. p-value: 0.5'

        "Plaintiff" only appears in plaintiff docs and has a frequency ratio of infinity.
        However, because json cannot represent infinity, we use -1 to represent infinity.
        >>> pl_index = statistical_analysis.vocabulary.index('plaintiff')
        >>> 'Plaintiff: Frequency Ratio: {}. p-value: {}'.format(freq[pl_index], freq_p[pl_index])
        'Plaintiff: Frequency Ratio: -1. p-value: 0.5'

        :rtype: tuple(list, list)
        """

        plaintiff_term_sums = np.array(self.dtm_count_plaintiff.sum(axis=0)).flatten()
        plaintiff_total = np.sum(plaintiff_term_sums)
        defendant_term_sums = np.array(self.dtm_count_defendant.sum(axis=0)).flatten()
        defendant_total = np.sum(defendant_term_sums)

        frequency_ratios = [0] * len(self.vocabulary)
        frequency_ratios_p = [''] * len(self.vocabulary)

        # probability for any given word to come from the plaintiff corpus
        p_plaintiff = float(plaintiff_total / (plaintiff_total + defendant_total))

        # Calculate frequency ratios and handle 0/infinity cases.
        for i in range(len(self.vocabulary)):
            if plaintiff_term_sums[i] == 0:
                frequency_ratios[i] = 0
            elif defendant_term_sums[i] == 0:
                # -1 represents infinity here because infinity can't be used in json
                frequency_ratios[i] = -1
            else:
                frequency_ratios[i] = ((plaintiff_term_sums[i] / plaintiff_total) /
                                       (defendant_term_sums[i] / defendant_total))

            p = binom_test(plaintiff_term_sums[i], plaintiff_term_sums[i] + defendant_term_sums[i],
                           p_plaintiff)
            frequency_ratios_p[i] = self._get_p_value_as_string(p)

        return frequency_ratios, frequency_ratios_p

    def mann_whitney_rho(self):
        """
        Calculates Mann-Whitney Rho scores for each term, which is a version of the Mann-Whitney U
        test normalized to return results between 0 and 1.

        The useful of this test to identify divergent terms is best explained by Ted Underwood.
        https://tedunderwood.com/2011/11/09/identifying-the-terms-that-characterize-an-author-or-genre-why-dunnings-may-not-be-the-best-method/


        >>> statistical_analysis = load_dummy_data()
        Loading six test documents
        >>> mwr, mwr_p = statistical_analysis.mann_whitney_rho()

        Within the dummy documents (see corpus.py, _load_test_corpus() function) most words appear
        in equal numbers in the plaintiff and defendant dummy documents
        Hence, the word "and" gets a Mann-Whitney Rho score of 0 and a p-value of 1.
        >>> and_index = statistical_analysis.vocabulary.index('and')
        >>> 'And: MWR: {}. p-value: {}'.format(mwr[and_index], mwr_p[and_index])
        'And: MWR: 0.5. p-value: 0.792'

        The only terms that show divergence are 'plaintiff' and 'defendant'
        >>> pl_index = statistical_analysis.vocabulary.index('plaintiff')
        >>> 'Plaintiff: MWR: {}. p-value: {}'.format(mwr[pl_index], mwr_p[pl_index])
        'Plaintiff: MWR: 0.8333333333333334. p-value: 0.197'

        :rtype: tuple(list, list)
        """

        mwr = np.zeros(len(self.vocabulary)).tolist()
        mwr_p = [''] * len(self.vocabulary)
        for i in range(len(self.vocabulary)):
            x = np.squeeze(np.asarray(self.dtm_tf_plaintiff[:, i].todense()))
            y = np.squeeze(np.asarray(self.dtm_tf_defendant[:, i].todense()))

            mwu = mwu_scipy(x, y, alternative='two-sided')

            mwr[i] = mwu.statistic / (len(x) * len(y))
            mwr_p[i] = self._get_p_value_as_string(mwu.pvalue)

        return mwr, mwr_p

    def dunning_log_likelihood(self):
        """
        This tests general distinctiveness, regardless of direction.
        See https://de.dariah.eu/tatom/feature_selection.html#chi2

        >>> statistical_analysis = load_dummy_data()
        Loading six test documents
        >>> dunning, dunning_p = statistical_analysis.dunning_log_likelihood()

        Within the dummy documents (see corpus.py, _load_test_corpus() function) most words appear
        in equal numbers in the plaintiff and defendant dummy documents
        Hence, the word "and" gets a log-likelihood score of 0 and a p-value of 1.
        >>> and_index = statistical_analysis.vocabulary.index('and')
        >>> 'And: Dunning: {}. p-value: {}'.format(dunning[and_index], dunning_p[and_index])
        'And: Dunning: 0.0. p-value: 1.0'

        The only terms that show divergence are 'plaintiff' and 'defendant'
        >>> pl_index = statistical_analysis.vocabulary.index('plaintiff')
        >>> 'Plaintiff: Dunning: {}. p-value: {}'.format(dunning[pl_index], dunning_p[pl_index])
        'Plaintiff: Dunning: 1.0464962875290957. p-value: 0.306'

        :rtype: tuple(list, list)
        """
        plaintiff_term_sums = np.array(self.dtm_count_plaintiff.sum(axis=0)).flatten()
        plaintiff_total = np.sum(plaintiff_term_sums)
        defendant_term_sums = np.array(self.dtm_count_defendant.sum(axis=0)).flatten()
        defendant_total = np.sum(defendant_term_sums)

        log_likelihoods = np.zeros(len(self.vocabulary))
        log_likelihoods_p = [''] * len(self.vocabulary)

        for i in range(len(self.vocabulary)):
            a = float(plaintiff_term_sums[i]) + 1
            b = float(defendant_term_sums[i]) + 1
            c = plaintiff_total
            d = defendant_total

            e1 = c * (a + b) / (c + d)
            e2 = d * (a + b) / (c + d)
            dunning_log_likelihood = 2 * (a * math.log(a / e1) + b * math.log(b/e2))

            if a*math.log(a / e1) < 0:
                dunning_log_likelihood = -dunning_log_likelihood

            log_likelihoods[i] = dunning_log_likelihood
            p = 1 - chi2.cdf(abs(dunning_log_likelihood), 1)
            log_likelihoods_p[i] = self._get_p_value_as_string(p)

        return log_likelihoods.tolist(), log_likelihoods_p

    def correlation_coefficient(self):
        """
        Calculate the correlation coefficients between all tokens and store the top 5 correlated
        1-grams.


        Sparse version of corrcoef adapted from:
        https://stackoverflow.com/questions/19231268/correlation-coefficients-for-sparse-matrix-in-python
        :return:
        """

        memory_gb_available = psutil.virtual_memory().available / 1024 / 1024 / 1024
        if memory_gb_available < 60:
            print("Calculating the correlation coefficients for all terms requires more memory",
                  "(at least 60 GB) than is available on this system. Since the correlations are",
                  "not essential, they will not be calculated here and you will instead see blank",
                  "spaces where they would otherwise appear.")
            return [""] * len(self.vocabulary)

        print("starting cor coeff calc")

        tf_transformer = TfidfTransformer(use_idf=False)
        dtm_tf_all = tf_transformer.fit_transform(self.dtm_count_all)
        dtm_tf_all = dtm_tf_all.T.tocsr()

        dtm_tf_all = self.dtm_tf_plaintiff.T.tocsr()
        from IPython import embed;embed()


#        A = A.astype(np.float64)
        n = dtm_tf_all.shape[1]

        # Compute the covariance matrix
        rowsum = dtm_tf_all.sum(1)
        centering = rowsum.dot(rowsum.T.conjugate()) / n
        C = (dtm_tf_all.dot(dtm_tf_all.T.conjugate()) - centering) / (n - 1)

        # The correlation coefficients are given by
        # C_{i,j} / sqrt(C_{i} * C_{j})
        d = np.diag(C)
        corr_coeffs = C / np.sqrt(np.outer(d, d))

        correlation_results = [''] * len(self.vocabulary)
        for i, token in enumerate(self.vocabulary):
            print(i)

            most_correlated_tokens = np.array(np.argsort(corr_coeffs[i, :])).flatten()[::-1]
            correlation_result = []
            for correlated_token_id in most_correlated_tokens:

                # store up to 5 results.
                if len(correlation_result) == 5:
                    break

                correlation = corr_coeffs[i, correlated_token_id]
                correlated_token = self.vocabulary[correlated_token_id]

                if np.isnan(correlation):
                    continue
                # only store correlated 1-grams
                elif len(correlated_token.split()) > 1:
                    continue
                # token cannot contain the correlated token (e.g. "said" cannot be correlated term
                # with "he said")
                elif token.find(correlated_token) > -1:
                    continue

                # skip he, she, s/he
                elif correlated_token in {'he', 'she', 's/he'}:
                    continue

                else:
                    correlation_result.append((correlated_token, correlation))

            correlation_results[i] = correlation_result

        return correlation_results

    @staticmethod
    def _get_p_value_as_string(p: float):
        """
        Returns p value as a string with cutoffs for p<0.0001, p<0.001, p<0.01, p<0.05, p<0.1.

        :rtype: str
        """

        if p < 0.0001:
            return '<0.0001'
        elif p < 0.001:
            return '<0.001'
        elif p < 0.01:
            return '<0.01'
        elif p < 0.05:
            return '<0.05'
        elif p < 0.1:
            return '<0.1'
        else:
            return f'{np.round(p, 3)}'


def load_dummy_data():
    """
    Loads a dummy dataset with six documents. Used mostly for the doc tests.

    :rtype: StatisticalAnalysis
    """
    from tobacco_litigation.corpus import Corpus
    from tobacco_litigation.dataset_creation import get_count_doc_term_matrices
    corpus = Corpus(use_test_corpus=True)
    vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant = get_count_doc_term_matrices(
        max_features=50000, min_df=2, corpus=corpus, part_of_speech=False)
    statistical_analysis = StatisticalAnalysis(dtm_count_all, dtm_count_plaintiff,
                                               dtm_count_defendant, vocabulary)
    return statistical_analysis


if __name__ == '__main__':
    a = load_dummy_data()
