import os
import pickle
import re
from pathlib import Path
import random

import nltk
import pandas
from IPython import embed

from tobacco_litigation.closing import Closing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from tobacco_litigation.configuration import CONTRACTIONS, WORD_SPLIT_REGEX, \
    WORD_SPLIT_PATTERN, STOP_WORDS_LITIGATION, NGRAM_RANGE

# we need shuffling but it should produce the same results every time.
random.seed(0)





class Corpus:

    def __init__(self, use_test_corpus=False):
        self.closings = self._load_closings()
        self.test_corpus=False
        if use_test_corpus:
            print("Loading six test documents")
            self.test_corpus=True
            self._load_test_corpus()


    def _load_closings(self):

        base_path = Path(os.path.abspath(os.path.dirname(__file__)))
        full_path = base_path.joinpath(Path('data', 'litigation_dataset_master.csv'))
        closings_metadata = pandas.read_csv(full_path)

        closings = []
        for id in range(len(closings_metadata)):
            # if id >= 3: break


            closing = Closing(case =                 closings_metadata['CASE'][id],
                              side =                 closings_metadata['SIDE'][id],
                              phase =                closings_metadata['PHASE'][id],
                              trial_date =           closings_metadata['TRIAL DATE'][id],
                              doc_date =             closings_metadata['DOC DATE'][id],
                              outcome =              closings_metadata['OUTCOME'][id],
                              defendants =           closings_metadata['DEFENDANTS'][id],
                              defendant_counsel =    closings_metadata['DEF COUNSEL'][id],
                              compensatory_damages = closings_metadata['Compensatory'][id],
                              punitive_damages =     closings_metadata['PUNITIVES'][id],
                              filename =             closings_metadata['FILENAME'][id],
                              tid =                  closings_metadata['tid'][id])
            closings.append(closing)

        return closings

    def _load_test_corpus(self):

        self.closings = self.closings[:6]
        data = [
            {'side': 'plaintiff', 'text': 'This is one sample plaintiff document.'},
            {'side': 'plaintiff', 'text': 'Here is another plaintiff document.'},
            {'side': 'plaintiff', 'text': ('No terms other than pl and def show any '
                                           'differences between pl and def docs.')},
            {'side': 'defendant', 'text': 'This is one sample defendant document.'},
            {'side': 'defendant', 'text': 'Here is another defendant document.'},
            {'side': 'defendant', 'text': ('No terms other than pl and def show any '
                                           'differences between pl and def docs.')},
        ]
        for i in range(6):
            self.closings[i].side = data[i]['side']
            self.closings[i].text_clean = data[i]['text']



    def document_iterator(self, side, part_of_speech=False, split_text_into_sections=False):
        """
        Provides a document iterator that yields one closing statement at a time
        Accepts 'plaintiff', 'defendant', and 'both' as side.

        split_text_into_slices=True yields each text in 100 word slices

        >>> from tobacco_litigation.corpus import Corpus
        >>> c = Corpus()
        >>> doc_iterator = c.document_iterator(side='both')
        >>> first_doc = next(doc_iterator)
        >>> first_doc[:52]
        'closing statement mr carter good afternoon you heard'


        split_text_into_slices=True yields each text in 100 word slices
        >>> section_iterator = c.document_iterator(side='both', split_text_into_sections=True)
        >>> first_section = next(section_iterator)
        >>> len(first_section.split())
        100

        """

        valid_sides = {'plaintiff', 'defendant', 'both'}
        if side not in valid_sides:
            raise ValueError(f"document_iterator accepts {valid_sides} as sides but not {side}.")

        for closing in self.closings:
            if side == 'both' or side == closing.side:
                if part_of_speech:
                    yield " ".join(closing.part_of_speech)
                else:
                    if split_text_into_sections:
                        text_clean_split = closing.text_clean.split()
                        sections = [" ".join(text_clean_split[i:i + 100]) for i in
                                    range(0, len(text_clean_split), 100)]
                        for section in sections:
                            yield section
                    else:
                        yield closing.text_clean

    def get_search_term_extracts(self, side, search_term, extract_size=100, no_passages=100):
        """
        Extracts passages with the search term from the documents, including up to extract_size
        surrounding characters on both sides.

        >>> from tobacco_litigation.corpus import Corpus
        >>> c = Corpus()
        >>> extracts = c.get_search_term_extracts(side='defendant', search_term='Proctor',
        ...                                       extract_size=40)
        >>> extracts[0]['text']
        'you can do about it and to use dr proctor s words game over unfortunately'

        :rtype: list
        """

        valid_sides = {'plaintiff', 'defendant', 'both'}
        if side not in valid_sides:
            raise ValueError(f"document_iterator accepts {valid_sides} as sides but not {side}.")

        if not isinstance(search_term, str):
            raise TypeError('search_term has to be a string, not {}'.format(type(search_term)))

        sections = []
        section_pattern = re.compile(r'\s.+\s', re.IGNORECASE | re.MULTILINE | re.DOTALL)

        for closing in self.closings:

            # if the side is not both and the side is not the same as the closing side, continue
            if side != 'both' and side != closing.side:
                continue

            current_pos = 0
            while True:
                match_pos = closing.text_clean.find(search_term.lower(), current_pos)
                if match_pos == -1:
                    break

                # original version had a curious try except Type Error here...
                section_raw = closing.text_clean[max(0, match_pos - extract_size): match_pos +
                                                         len(search_term) + extract_size]
                section = section_pattern.search(section_raw).group()
                section_text = " ".join(section.split())
                sections.append({
                    'tid': closing.tid,
                    'case': closing.case,
                    'side': side,
                    'date': closing.doc_date,
                    'text': section_text
                })

                current_pos = match_pos + len(search_term) + extract_size + 1

        random.shuffle(sections)
        sections = sections[:no_passages]
        sections.sort(key=lambda k: (k['case'], k['text']))

        return sections


if __name__ == '__main__':

    c = Closing('c', 'plaintiff', 1, '1998', '1987', 'w', 'RJR', 'test', 100, 100,
                'ahrens1_1_c_d.txt', 'otehu')

    corpus = Corpus()
    ex = corpus.get_search_term_extracts('plaintiff', 'Proctor')
    embed()
