import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow

from textgenrnn import textgenrnn
from tobacco_litigation.corpus import LitigationCorpus
from IPython import embed

def text_gen_rnn():

    corpus = LitigationCorpus()
    texts = [text for text in corpus.document_iterator('plaintiff')]

    texts = texts

    textgen = textgenrnn()

    while True:

        textgen.train_on_texts(texts=texts, num_epochs=3, gen_epochs=3,
                               train_size=0.8, new_model=False,
                               verbose=2)

        embed()


if __name__ == '__main__':

    text_gen_rnn()