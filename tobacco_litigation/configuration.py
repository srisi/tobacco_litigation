"""
This file contains the general configurations for the tobacco_litigation project.
This includes all stop words added to the sklearn set and all contractions.
"""

import os
import re
from pathlib import Path


BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))

# Split strings into tokens (includes splitting dashes)
WORD_SPLIT_PATTERN = r"\b\w+\b"
WORD_SPLIT_REGEX = re.compile(WORD_SPLIT_PATTERN)

NGRAM_RANGE = (1,10)

# retain only numbers between 1800 and 2050 because they can indicate years.
STOP_WORDS_LITIGATION = frozenset([str(i) for i in range(1800)] +
                                  [str(i) for i in range(2050, 10000)] +
                                  ["0" + str(i) for i in range(0, 10000)] +
                                  # terms on the bottom of each page of shulman1_0_c_d
                                  ['9dc6a7f69111', '32fa', '4b55aaf6'])

# exclude lawyer names because they don't contain helpful information
STOP_WORDS_NAMES = ['tullo', 'aycock', 'ingraham', 'barbanell', 'shirley', 'wilbert', 'cooper',
                    'piendle', 'mr', 'mrs', 'blasco', 'collar', 'kaplan', 'owens', 'dion',
                    'mccoy', 'clayton', 'jewett', 'bowman', 'dupre', 'hackimer', 'taylor',
                    'sammarco', 'ms', 'jordan', 'clay', '32fa', '4b55aaf6', '9dc6a7f69111',
                    'mccray', 'cooper', 'tate', 'ellen', 'cuba', 'glover', 'shapiro', 'enochs',
                    'heller', 'buck', 'haldeman', 'jewett', 'burney', 'tullo']

# contractions to expand in all documents
CONTRACTIONS = {
    "10 percent": 'ten percent',
    '20 percent': 'twenty percent',
    '30 percent': 'thirty percent',
    '40 percent': 'forty percent',
    '50 percent': 'fifty percent',
    '60 percent': 'sixty percent',
    '70 percent': 'seventy percent',
    '80 percent': 'eighty percent',
    '90 percent': 'ninety percent',
    '100 percent': 'one hundred percent',
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}
