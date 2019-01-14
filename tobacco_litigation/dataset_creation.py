import json
import re
import sqlite3
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer

from tobacco_litigation.configuration import WORD_SPLIT_PATTERN, STOP_WORDS_LITIGATION, \
    NGRAM_RANGE, STOP_WORDS_SKLEARN
from tobacco_litigation.corpus import LitigationCorpus
from tobacco_litigation.stats import StatisticalAnalysis


def create_json_dataset():
    """
    Creates the main json dataset that will be used to display the litigation data
    on www.tobacco-analytics.org/litigation

    Note: There are only two part of speech tags that are currently stored in the dataset:
    'Verb, Past Tense',     consisting of VBD
    'Verb, Present Tense',  consisting of VBP and VBZ

    The data is stored as a list of tuples, with each tuples having the following values:
    [ 0]    Token
    [ 1]    ngram                   int     Range: (1-10)
    [ 2]    is_pos                  int     1 if True (the token is a part of speech tag) else 0
    [ 3]    pos                     str     The part of speech representation of the token
    [ 4]    includes_frequent_term  int     1 if True (the token includes a frequent term) else 0
    [ 5]    count_all               int     Number of times the token appears in the corpus
    [ 6]    count_plaintiff         int     Number of times the token appears in plaintiff docs
    [ 7]    count_defense           int     Number of times the token appears in defendant docs
    [ 8]    freq_all                str     "y.yyyy%" Token frequency in the corpus
    [ 9]    freq_plaintiff          str     "y.yyyy%" Token frequency in plaintiff docs
    [10]    freq_defendant          str     "y.yyyy%" Token frequency in defendant docs
    [11]    freq_ratio              float   Frequency ratio (plaintiff/defendant) of the token
    [12]    freq_ratio_p            str     p-value of the frequency ratio, e.g. "p<0.01"
    [13]    dunning                 float   Dunning Log-Likelihood score of the token
    [14]    dunning_p               str     p-value of the Dunning Log-Likelihood score
    [14]    mwr                     float   Mann-Whitney Rho Score of the token
    [15]    mwr_p                   str     p-value of the Mann-Whitney Rho Score, e.g. "p<0.01"
    [16]    correlated_terms        list    List of the 1-grams most correlated with the token

    """

    # Create the datasets with and without POS
    create_distinctive_terms_dataset(part_of_speech=False)
    create_distinctive_terms_dataset(part_of_speech=True)


    # Gather all tokens and store them as a json
    con = sqlite3.connect('data/distinctive_tokens.db')
    cur = con.cursor()
    cur.execute('''SELECT 
                          token, ngram, is_pos, pos, includes_frequent_term, 
                          count_all, count_plaintiff, count_defendant, 
                          freq_all, freq_plaintiff, freq_defendant, 
                          freq_ratio, freq_ratio_p, dunning, dunning_p, mwr, mwr_p, 
                          correlated_terms 
                          
                   FROM distinctive_terms''')
    data = []
    for row in cur.fetchall():
        row = list(row)

        # Correlations are not calculated for the synthetic terms. In those cases, we
        # store a placeholder.
        try:
            # The correlated terms list is stored as a str -> parse to list
            correlated_terms = eval(row[17])
            row[17] = [(cor[0], round(cor[1], 3)) for cor in correlated_terms]
        except SyntaxError:
            row[17] = [('n/a', 0.0) for _ in range(5)]

        data.append(row)

    with open('data/distinctive_tokens.json', 'w') as out:
        json.dump(data, out)


def create_distinctive_terms_dataset(part_of_speech):
    """
    Generates the distinctive terms dataset and stores the results in distinctive_tokens.db
    For each token, calculates:
    - Frequency Ratio (How much more often does the token appear in the plaintiff docs?)
    - Dunning's Log-Likehood (How distinctive is the term for plaintiffs or defendants?)
    - Mann-Whitney Rho (How consistently is the token associated with plaintiffs or defendants?)

    :param part_of_speech: If True, use POS. Else, use the text of the closings
    """

    corpus = LitigationCorpus()

    if part_of_speech:
        max_features = 10000
        min_df = 50
    else:
        max_features = 50000
        min_df = 1

    vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant = get_count_doc_term_matrices(
        max_features, min_df, corpus, part_of_speech)

    print("Finished tokenizing")
    print("\nDistinctive terms matrices")
    print(f"All. Shape: {dtm_count_all.shape}. Count: {dtm_count_all.sum()}")
    print(f"Plaintiff. Shape: {dtm_count_plaintiff.shape}. Count: {dtm_count_plaintiff.sum()}")
    print(f"Defendant. Shape: {dtm_count_defendant.shape}. Count: {dtm_count_defendant.sum()}")

#    from IPython import embed;embed()

    # Run the statistical analyses for the corpus
    stats = StatisticalAnalysis(dtm_count_all, dtm_count_plaintiff, dtm_count_defendant, vocabulary)
    frequency_ratio, frequency_ratio_p = stats.frequency_score()
    mann_whitney_rho, mann_whitney_rho_p = stats.mann_whitney_rho()
    dunning_log_likelihood, dunning_log_likelihood_p = stats.dunning_log_likelihood()

    if not part_of_speech:
        # Store a dict that maps from token to pos
        text_to_pos_dict = get_text_to_pos_dict(corpus)
        # Store the
        with open('data/vocabulary.txt', 'w') as outfile:
            outfile.writelines([i + '\n' for i in vocabulary])

        # Calculate term correlations using sections of 100 words (for correlation calculations,
        # we only care about terms that appear in close proximity.)
        section_vocabulary, section_dtm_count_all, section_dtm_count_plaintiff, \
        section_dtm_count_defendant = get_count_doc_term_matrices(max_features, min_df, corpus,
                                          part_of_speech, split_text_into_sections=True)
        stats_sections = StatisticalAnalysis(section_dtm_count_all, section_dtm_count_plaintiff,
                                             section_dtm_count_defendant, section_vocabulary)
        term_correlations = stats_sections.correlation_coefficient()

    # Aggregate data before storing the results
    plaintiff_term_sums = np.array(dtm_count_plaintiff.sum(axis=0)).flatten()
    defendant_term_sums = np.array(dtm_count_defendant.sum(axis=0)).flatten()
    all_term_sums = np.array(dtm_count_all.sum(axis=0)).flatten()

    total_plaintiff = dtm_count_plaintiff.sum()
    total_defendant = dtm_count_defendant.sum()
    total_all = dtm_count_all.sum()

    # Store the results in a list, which can then be inserted into distinctive_tokens.db
    results = []
    for i in range(len(vocabulary)):

        # for POS, only retain past and present verbs.
        if part_of_speech and not vocabulary[i] in ['Verb, Present Tense', 'Verb, Past Tense']:
            continue

        token = {}
        token['token'] = vocabulary[i].upper() if part_of_speech else vocabulary[i]
        token['ngram'] = len(vocabulary[i].split())
        token['is_pos'] = 1 if part_of_speech else 0
        token['pos'] = '' if part_of_speech else text_to_pos_dict[vocabulary[i]]
        token['count_all'] = int(all_term_sums[i])
        token['count_plaintiff'] = int(plaintiff_term_sums[i])
        token['count_defendant'] = int(defendant_term_sums[i])
        token['freq_all'] = "{}%".format(np.round(all_term_sums[i] / total_all * 100, 8))
        token['freq_plaintiff'] = "{}%".format(np.round(plaintiff_term_sums[i] /
                                                        total_plaintiff * 100, 8))
        token['freq_defendant'] = "{}%".format(np.round(defendant_term_sums[i] /
                                                        total_defendant * 100, 8))

        token['freq_ratio'] = np.round(frequency_ratio[i], 4)
        token['freq_ratio_p'] = frequency_ratio_p[i]
        token['dunning'] = np.round(dunning_log_likelihood[i], 4)
        token['dunning_p'] = dunning_log_likelihood_p[i]
        token['mwr'] = np.round(mann_whitney_rho[i], 3)
        token['mwr_p'] = mann_whitney_rho_p[i]

        token['correlated_terms'] = ''
        if not part_of_speech:
            try:
                token['correlated_terms'] = str(term_correlations[token['token']])
            except KeyError:
                print(f"Correlated tokens for {token['token']} not available.")

        # Check if the token includes a frequent term (only for text, not pos)
        token['includes_frequent_term'] = 0
        if not part_of_speech:
            for t in vocabulary[i].split():
                if t in STOP_WORDS_SKLEARN:
                    token['includes_frequent_term'] = 1
                    break

        results.append(token)

    # Sort tokens by count
    results = sorted(results, key=lambda k: k['count_all'], reverse=True)

    # Store results into distinctive_tokens.db
    create_token_database()
    con = sqlite3.connect('data/distinctive_tokens.db')
    cur = con.cursor()

    for t in results:
        placeholder = ", ".join(["?"] * len(t))
        sql_insert = f'REPLACE INTO distinctive_terms({",".join(t.keys())}) VALUES({placeholder});'
        cur.execute(sql_insert, list(t.values()))
    con.commit()


def get_count_doc_term_matrices(max_features: int, min_df: float, corpus: LitigationCorpus,
                                part_of_speech: bool, split_text_into_sections=False):
    """
    Creates the count (not term frequency) document term matrices

    :param max_features:
    :param min_df:
    :param corpus:
    :param part_of_speech:
    """
    if corpus.test_corpus:
        ngram_range = (1, 1)
    else:
        ngram_range = NGRAM_RANGE

    vectorizer_all = CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                                     stop_words=STOP_WORDS_LITIGATION, min_df=min_df,
                                     token_pattern=WORD_SPLIT_PATTERN)

    docs_all_iterator = corpus.document_iterator(side='both', part_of_speech=part_of_speech,
                                                 split_text_into_sections=split_text_into_sections)
    dtm_count_all = vectorizer_all.fit_transform(docs_all_iterator)
    vocabulary = vectorizer_all.get_feature_names()

    count_vectorizer_sides = CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                                             stop_words=STOP_WORDS_LITIGATION,
                                             token_pattern=WORD_SPLIT_PATTERN,
                                             vocabulary=vocabulary)
    docs_plaintiff_iterator = corpus.document_iterator(side='plaintiff',
                   part_of_speech=part_of_speech, split_text_into_sections=split_text_into_sections)
    docs_defendant_iterator = corpus.document_iterator(side='defendant',
                   part_of_speech=part_of_speech, split_text_into_sections=split_text_into_sections)
    dtm_count_plaintiff = count_vectorizer_sides.transform(docs_plaintiff_iterator)
    dtm_count_defendant = count_vectorizer_sides.transform(docs_defendant_iterator)

    # add all of the synthetic terms to the vocabulary and the dtms
    vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant = add_synthetic_terms(
        corpus, vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant, part_of_speech
    )

    return vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant


def get_text_to_pos_dict(corpus):

    # POS requires capitalization. These lines create a dict of tokens to pos that takes capitalization
    # into account

    text_to_pos_dict = defaultdict(lambda: '')

    # vectorize all documents, taking capitalization into account
    pos_vectorizer = CountVectorizer(max_features=50000, ngram_range=NGRAM_RANGE,
                                     stop_words=STOP_WORDS_LITIGATION, min_df=1,
                                     token_pattern=WORD_SPLIT_PATTERN, lowercase=False)
    docs_all_iterator = corpus.document_iterator(side='both', part_of_speech=False)
    dtm_pos = pos_vectorizer.fit_transform(docs_all_iterator)
    # sort vocabulary from least to most frequent
    totals = np.array(dtm_pos.sum(axis=0))[0]
    pos_vocabulary = pos_vectorizer.get_feature_names()
    sorted_pos_vocabulary = [token for (total, token) in sorted(zip(totals, pos_vocabulary))]
    # add all tokens from less frequent (The house) to more frequent (the house)
    for token in sorted_pos_vocabulary:
        token_pos = nltk.pos_tag([token])[0][1]
        text_to_pos_dict[token.lower()] = token_pos

    return text_to_pos_dict


def add_synthetic_terms(corpus, vocabulary, dtm_count_all, dtm_count_plaintiff,
                        dtm_count_defendant, part_of_speech):
    """
    This function adds all of the synthetic terms to the document term matrices and the vocabulary

    The dataset uses a number of synthetic terms:
    - he and she get aggregated as s/he to remove patterns caused by plaintiff gender
    - mr, ms, mrs get aggregated for the same reason.
    - the same is done for relatives, e.g. mother and father as father/mother

    Beyond that, a few very similar terms are aggregated so their results can be combined and don't
    need to be individually reported:
    - smoking is/was dangerous
    - decision/s
    - risk/s
    - warning/s

    - customer/s
    - teenager/s
    - kid/s

    - specific/ally

    """


    # if we use dummy data, don't add the synthetic terms
    if corpus.test_corpus:
        return vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant

    # For part of speech, only add "verb past tense" and "verb present tense"
    if part_of_speech:

        # add summary for verb present and past tense.
        additions_all = hstack([
            (dtm_count_all[:, vocabulary.index('vbd')]).tocoo(),
            (dtm_count_all[:, vocabulary.index('vbp')] +
             dtm_count_all[:, vocabulary.index('vbz')]).tocoo()
        ])
        additions_plaintiff = hstack([
            (dtm_count_plaintiff[:, vocabulary.index('vbd')]).tocoo(),
            (dtm_count_plaintiff[:, vocabulary.index('vbp')] +
             dtm_count_plaintiff[:, vocabulary.index('vbz')]).tocoo()
        ])
        additions_defendant = hstack([
            (dtm_count_defendant[:, vocabulary.index('vbd')]).tocoo(),
            (dtm_count_defendant[:, vocabulary.index('vbp')] +
             dtm_count_defendant[:, vocabulary.index('vbz')]).tocoo()
        ])
        dtm_count_all = hstack([dtm_count_all, additions_all]).tocsr()
        dtm_count_plaintiff = hstack([dtm_count_plaintiff, additions_plaintiff]).tocsr()
        dtm_count_defendant = hstack([dtm_count_defendant, additions_defendant]).tocsr()

        vocabulary.append('Verb, Past Tense')
        vocabulary.append('Verb, Present Tense')
    
    else:

        # initialize additions by adding mr/mrs/ms
        additions_all = (dtm_count_all[:, vocabulary.index('mr')]  + 
                         dtm_count_all[:, vocabulary.index('mrs')] + 
                         dtm_count_all[:, vocabulary.index('mrs')]).tocoo()
        additions_plaintiff = (dtm_count_plaintiff[:, vocabulary.index('mr')] +
                               dtm_count_plaintiff[:, vocabulary.index('mrs')] +
                               dtm_count_plaintiff[:, vocabulary.index('ms')]).tocoo()
        additions_defendant = (dtm_count_defendant[:, vocabulary.index('mr')] + 
                               dtm_count_defendant[:, vocabulary.index('mrs')] +
                               dtm_count_defendant[:, vocabulary.index('ms')]).tocoo()
        vocabulary.append('mr/mrs/ms')

        additions = [
            ('husband/wife', ['husband', 'wife']),
            ('brother/sister', ['brother', 'sister']),
            ('father/mother', ['father', 'mother']),

            ('smoking is/was dangerous', ['smoking is dangerous', 'smoking was dangerous']),

            ('customer/s', ['customer', 'customers']),
            ('teenager/s', ['teenager', 'teenagers']),
            ('kid/s', ['kid', 'kids']),

            ('specific/ally', ['specific', 'specifically']),
            ('decision/s', ['decision', 'decisions']),
            ('risk/s', ['risk', 'risks']),
            ('warning/s', ['warning', 'warnings']),

            ('RELATIVES', ['husband', 'wife', 'brother', 'brothers', 'sister', 'sisters',
                           'father', 'mother', 'grandfather', 'grandmother',
                           'daughter', 'daughters', 'granddaughter', 'son', 'sons', 'grandson',
                           'uncle', 'aunt']),
        ]

        for addition in additions:
            synthetic_vector_all = None
            synthetic_vector_plaintiff = None
            synthetic_vector_defendant = None
            for term in addition[1]:
                idx = vocabulary.index(term)
                if synthetic_vector_all == None:
                    synthetic_vector_all = dtm_count_all[:, idx]
                else:
                    synthetic_vector_all += dtm_count_all[:, idx]
                if synthetic_vector_plaintiff == None:
                    synthetic_vector_plaintiff = dtm_count_plaintiff[:, idx]
                else:
                    synthetic_vector_plaintiff += dtm_count_plaintiff[:, idx]
                if synthetic_vector_defendant == None:
                    synthetic_vector_defendant = dtm_count_defendant[:, idx]
                else:
                    synthetic_vector_defendant += dtm_count_defendant[:, idx]

            additions_all = hstack([additions_all, synthetic_vector_all])
            additions_plaintiff = hstack([additions_plaintiff, synthetic_vector_plaintiff])
            additions_defendant = hstack([additions_defendant, synthetic_vector_defendant])
            vocabulary.append(addition[0])

        # create s/he (he or she) synthetic terms
        for idx, term in enumerate(vocabulary):
            try:
                if 'he' in term.split():
                    synthetic_term = re.sub(r'\bhe\b', 's/he', term)
                    idx2 = vocabulary.index(re.sub(r'\bhe\b', 'she', term))
                    vocabulary.append(synthetic_term)
                    synthetic_vector_all = dtm_count_all[:, idx] + dtm_count_all[:, idx2]
                    synthetic_vector_plaintiff = (dtm_count_plaintiff[:, idx] +
                                                  dtm_count_plaintiff[:, idx2])
                    synthetic_vector_defendant = (dtm_count_defendant[:, idx] +
                                                  dtm_count_defendant[:, idx2])

                    additions_all = hstack([additions_all, synthetic_vector_all])
                    additions_plaintiff = hstack([additions_plaintiff, synthetic_vector_plaintiff])
                    additions_defendant = hstack([additions_defendant, synthetic_vector_defendant])

            # some term exist only for "he"
            except (ValueError, AttributeError):
                pass

        dtm_count_all = hstack([dtm_count_all, additions_all]).tocsr()
        dtm_count_plaintiff = hstack([dtm_count_plaintiff, additions_plaintiff]).tocsr()
        dtm_count_defendant = hstack([dtm_count_defendant, additions_defendant]).tocsr()
        
    return vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant

def create_token_database():
    """
    Initializes the distinctive_tokens database

    """

    con = sqlite3.connect('data/distinctive_tokens.db')
    cur = con.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS distinctive_terms (
                token   text NOT NULL,
                ngram   INT  NOT NULL,
                is_pos  INT  NOT NULL,
                pos     text,
                includes_frequent_term INT,

                count_all       INT NOT NULL,
                count_plaintiff INT,
                count_defendant INT,
                freq_all        text NOT NULL,
                freq_plaintiff  text NOT NULL,
                freq_defendant  text NOT NULL,

                freq_ratio      REAL NOT NULL,
                freq_ratio_p    text NOT NULL,
                dunning         REAL  NOT NULL,
                dunning_p       text  NOT NULL,
                mwr             REAL  NOT NULL,
                mwr_p           text  NOT NULL,
                
                correlated_terms text,
                footnote  text,


                UNIQUE(token) ON CONFLICT REPLACE );
      ''')


if __name__ == '__main__':

    create_json_dataset()
