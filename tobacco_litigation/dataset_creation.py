import re
import sqlite3
from collections import defaultdict

import nltk
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer

from tobacco_litigation.configuration import WORD_SPLIT_PATTERN, STOP_WORDS_LITIGATION, NGRAM_RANGE
from tobacco_litigation.corpus import Corpus
from tobacco_litigation.stats import StatisticalAnalysis


def create_json_dataset():

    create_distinctive_terms_dataset(part_of_speech=False)
    create_distinctive_terms_dataset(part_of_speech=True)



def create_distinctive_terms_dataset(part_of_speech):

    corpus = Corpus()

    if part_of_speech:
        max_features = 10000
        min_df = 50
    else:
        max_features = 50000
        min_df = 1

    # print("Creating section doc-term-matrices to calculate token correlations.")
    # voc, section_dtm_count_all, section_dtm_count_plaintiff, section_dtm_count_defendant = \
    #     get_count_doc_term_matrices(max_features, min_df, corpus, part_of_speech,
    #                                 split_text_into_sections=True)
    # stats_sections = StatisticalAnalysis(section_dtm_count_all, section_dtm_count_plaintiff,
    #                                      section_dtm_count_defendant, voc)
    # term_correlations = stats_sections.correlation_coefficient()

    vocabulary, dtm_count_all, dtm_count_plaintiff, dtm_count_defendant = get_count_doc_term_matrices(
        max_features, min_df, corpus, part_of_speech)

    print("Finished tokenizing")
    print("\nDistinctive terms matrices")
    print(f"All. Shape: {dtm_count_all.shape}. Count: {dtm_count_all.sum()}")
    print(f"Plaintiff. Shape: {dtm_count_plaintiff.shape}. Count: {dtm_count_plaintiff.sum()}")
    print(f"Defendant. Shape: {dtm_count_defendant.shape}. Count: {dtm_count_defendant.sum()}")

    stats = StatisticalAnalysis(dtm_count_all, dtm_count_plaintiff, dtm_count_defendant, vocabulary)
    frequency_ratio, frequency_ratio_p = stats.frequency_ratio()
    mann_whitney_rho, mann_whitney_rho_p = stats.mann_whitney_rho()
    dunning_log_likelihood, dunning_log_likelihood_p = stats.dunning_log_likelihood()

    if not part_of_speech:
        text_to_pos_dict = get_text_to_pos_dict(corpus)
        with open('data/vocabulary.txt', 'w') as outfile:
            outfile.writelines([i + '\n' for i in vocabulary])

        print("Creating section doc-term-matrices to calculate token correlations.")
        _, section_dtm_count_all, section_dtm_count_plaintiff, section_dtm_count_defendant = \
            get_count_doc_term_matrices(max_features, min_df, corpus, part_of_speech,
                                        split_text_into_sections=True)
        stats_sections = StatisticalAnalysis(section_dtm_count_all, section_dtm_count_plaintiff,
                                             section_dtm_count_defendant, vocabulary)
        term_correlations = stats_sections.correlation_coefficient()

    plaintiff_term_sums = np.array(dtm_count_plaintiff.sum(axis=0)).flatten()
    defendant_term_sums = np.array(dtm_count_defendant.sum(axis=0)).flatten()
    all_term_sums = np.array(dtm_count_all.sum(axis=0)).flatten()

    total_plaintiff = dtm_count_plaintiff.sum()
    total_defendant = dtm_count_defendant.sum()
    total_all = dtm_count_all.sum()

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
        token['freq_all'] = "{}%".format(np.round(all_term_sums[i] / total_all * 100, 3))
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

        token['correlated_terms'] = '' if part_of_speech else str(term_correlations[i])

        # Check if the token includes a frequent term (only for text, not pos)
        token['includes_frequent_term'] = 0
        if not part_of_speech:
            token['includes_frequent_term'] = 0
            for t in vocabulary[i].split():
                if t in STOP_WORDS_LITIGATION:
                    token['includes_frequent_term'] = 1
                    break

        results.append(token)

    results = sorted(results, key=lambda k: k['count_all'], reverse=True)

    create_token_database()
    con = sqlite3.connect('data/distinctive_tokens.db')
    cur = con.cursor()

    for t in results:
        placeholder = ", ".join(["?"] * len(t))
        sql_insert = f'INSERT INTO distinctive_terms({",".join(t.keys())}) VALUES({placeholder});'
        cur.execute(sql_insert, list(t.values()))
    con.commit()


def get_count_doc_term_matrices(max_features: int, min_df: float, corpus: Corpus,
                                part_of_speech: bool, split_text_into_sections=False):
    """
    Creates the count (not term frequency) document term matrices

    :param max_features:
    :param min_df:
    :param corpus:
    :param part_of_speech:
    :return:
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
    # 4/3/17: why, then, is this only active when POS is not. shouldn't it be if part_of_speech:...
    # 4/18/17: re previous comment: This is where the pos of each text token is retrieved.
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

    con = sqlite3.connect('data/distinctive_tokens.db')
    cur = con.cursor()

    # token -> what token (pos and text)
    # ngram -> ngram
    # is_pos -> is this a part of speech token (1) or not (0)
    # pos -> the part of speech equivalent of the token (if text)
    # includes_frequent_term -> (text only) 1 if it includes a frequent term, 0 otherwise


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
#    create_distinctive_terms_dataset(part_of_speech=False)
    create_json_dataset()