from tobacco_litigation.configuration import BASE_PATH
import csv
from pathlib import Path
from IPython import embed
from collections import defaultdict


def create_liwc_category_term_lists():
    """
    Returns a dictionary that maps every liwc category to a list of terms contained in it.
    >>> liwc_cat = create_liwc_category_term_lists()
    >>> liwc_cat['Nonfluencies']
    ['well', 'oh', 'sighed', 'sigh', 'um', 'ah']

    :return:
    """

    category_term_lists = defaultdict(list)

    with open (Path(BASE_PATH, 'data', 'liwc_terms_to_category.csv')) as csv_file:
        for row in csv.DictReader(csv_file):
            term = row['Word']
            for category, val in row.items():
                if category == 'Word':
                    continue
                elif val == 'X':
                    category = LWIC_ABREV_TO_CAT[category]
                    category_term_lists[category].append(term)
                else:
                    pass

    return category_term_lists


LWIC_ABREV_TO_CAT = {
    'function': 'Total Function Words',
    'pronoun': 'Total Pronouns',
    'ppron': 'Presonal Pronouns',
    'i': '1st Person Singular',
    'we': '1st Person Plural',
    'you': '2nd Person',
    'shehe': '3rd Person Singular',
    'they': '3rd Person Plural',

    'ipron': 'Impersonal Pronouns',
    'article': 'Articles',
    'prep': 'Prepositions',
    'auxverb': 'Auxiliary Verbs',
    'adverb': 'Common Adverbs',
    'conj': 'Conjunctions',
    'negate': 'Negations',

    'verb': 'Common Verbs',
    'adj': 'Common Adjectives',
    'compare': 'Comparisons',
    'interrog': 'Interrogatives',
    'number': 'Numbers',
    'quant': 'Quantifiers',

    'affect': 'Affective Processes',
    'posemo': 'Positive Emotion',
    'negemo': 'Negative Emotion',
    'anx': 'Anxiety',
    'anger': 'Anger',
    'sad': 'Sadness',

    'social': 'Social Processes',
    'family': 'Family',
    'friend': 'Friends',
    'female': 'Female References',
    'male': 'Male References',

    'cogproc': 'Cognitive Processes',
    'insight': 'Insight',
    'cause': 'Causation',
    'discrep': 'Discrepancy',
    'tentat': 'Tentative',
    'certain': 'Certainty',
    'differ': 'Differentiation',

    'percept': 'Perceptual Processes',
    'see': 'See',
    'hear': 'Hear',
    'feel': 'Feel',

    'bio': 'Biological Processes',
    'body': 'Body',
    'health': 'Health',
    'sexual': 'Sexual',
    'ingest': 'Ingestion',

    'drives': 'Drives',
    'affiliation': 'Affiliation',
    'achieve': 'Achievement',
    'power': 'Power',
    'reward': 'Reward',
    'risk': 'Risk',

    'focuspast': 'Past Focus',
    'focuspresent': 'Present Focus',
    'focusfuture': 'Future Focus',

    'relativ': 'Relativity',
    'motion': 'Motion',
    'space': 'Space',
    'time': 'Time',

    'work': 'Work',
    'leisure': 'Leisure',
    'home': 'Home',
    'money': 'Money',
    'relig': 'Religion',
    'death': 'Death',

    'informal': 'Informal Language',
    'swear': 'Swear Words',
    'netspeak': 'Netspeak',
    'assent': 'Assent',
    'nonflu': 'Nonfluencies',
    'filler': 'Fillers'
}

if __name__ == '__main__':
    create_liwc_category_term_lists()