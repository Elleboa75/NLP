from collections import defaultdict
import matplotlib
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict
d = defaultdict(float)
d["new key"]

PTB_FILES = {'train': 'data/sec02-21.gold.tagged', 'dev': 'data/sec00.gold.tagged'}
TED_FILES = {'train': 'data/ted-train.txt', 'test': 'data/ted-test.txt'}


# making a shortcut for the read-only get for defaultdict that supports several keys
# i.e. defget(d, [k1, k2]) will return the value of d[k1][k2] without altering d
def defget(d, keys):
    for k in keys:
        d = d.get(k, d.default_factory() if isinstance(d, defaultdict) else 0)
    return d

# testing if it works as intended
d = defaultdict(lambda: defaultdict(int))
d[1][1] = 2
d[2][1] = 3
print(d)
print(defget(d, [1, 1]))
print(defget(d, [3]))
print(defget(d, [3, 1]))
print(d)


def read_data(fname, h=1, max_lines=np.inf):
    """
    Reads in the data from a file and returns a vocabulary as a set
    and the sentence-padded data as a list of list.

    :param fname: path to the file
    :param max_lines: the number of top lines to read (can be used for debugging)
    :param h: the length of n-gram history
    :returns: data as a list of lists and vocabulary as a set
    """
    data = []
    start = h * ["<s>"]
    end = ["</s>"]

    with open(fname, "r", encoding='UTF-8') as F:
        for k, line in enumerate(F):
            # an optional cut-off to read a part of the data
            if k > max_lines:
                break
            words = line.strip().split()
            # padding the sentence
            sent = start + words + end #also adding the ending for unigrams to capture most likley stopping time when generating the sentences
            data.append(sent)
    return data




def train_ngram(data, N=2, k=0):
    """
    Trains an n-gram language model with optional add-k smoothing
    and additionally returns the unigram model

    :param data: text-data as returned by the pre-defined function read_data
    :param N: (N>1) the order of the ngram e.g. N=2 gives a bigram
    :param k: optional add-k smoothing
    :returns: ngram and unigram
    """
    # ngram[history][word] = #(history,word)
    # for history of length >1, join tokens with a singel white space (see tests below)

    ## YOUR CODE HERE ##
    ## you can have auxiliary functions if needed, inside this function

    ngram_counts = defaultdict(Counter)
    unigram_counts = Counter()
    total_tokens = 0

    vocab = set()
    for sent in data:
        for i in range(N - 1, len(sent)):
            token = sent[i]
            unigram_counts[token] += 1
            total_tokens += 1
            vocab.add(token)


            history = " ".join(sent[i-(N-1):i])
            ngram_counts[history][token] += 1
    vocab_size = len(vocab)

    unigram = defaultdict(float)

    for token, count in unigram_counts.items():
        unigram[token] = count / total_tokens
    ngram = defaultdict(lambda: unigram)

    for history, counts in ngram_counts.items():
        history_total = sum(counts.values())

        denominator = history_total + (k * vocab_size)
        if denominator > 0:
            default_probability = k/denominator
        else:
            default_probability = 0.0
        #default_probability = k / denominator if denominator > 0 else 0.0
        print(default_probability, " ", denominator)
        inner_distribution = defaultdict(lambda: default_probability)

        for token, count in counts.items():
            inner_distribution[token] = (count + k) / denominator

        ngram[history] = inner_distribution
    print(ngram, " ", unigram)
    return ngram, unigram

# This shouldn't take more than 2min
Data1 = read_data(TED_FILES["train"], h = 1) ## YOUR CODE HERE ##
# use the global variable for the file path
# non-smoothed bigram
Bigram, Unigram = train_ngram(Data1, N=2, k=0)
# smoothed bigram
Bigram_sm, Unigram_sm = train_ngram(Data1, N=2, k=1)


Data2 = read_data(TED_FILES["train"], h = 2) ## YOUR CODE HERE ##
# non-smoothed trigram
Trigram, Unigram_ = train_ngram(Data2, N=3, k=0)
# smoothed trigram
Trigram_sm, Unigram_sm_ = train_ngram(Data2, N=3, k=1)

print (Unigram_sm['rainbow-unicorn'])
#assert defget(Bigram_sm, ['the', 'Uns33n']) != defget(Bigram_sm, ['a', 'Uns33n'])
#assert defget(Trigram, ['all all', 'unicorns']) == 0