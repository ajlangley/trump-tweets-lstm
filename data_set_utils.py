import nltk
import numpy as np
import itertools

START = '<START>'
END = '<END>'

def build_training_set(filename):
    with open(filename, 'r') as infile:
        raw_tweets = [tweet.replace('&amp;', '&') for tweet in infile.readlines()]
        tweets = [[START] + [c for c in tweet if c != '\n'] + [END]
            for tweet in raw_tweets]
        index_to_char = get_chars(tweets)
        char_to_index = dict([(w, i) for i, w in enumerate(index_to_char)])
        tokenized_tweets = [[char_to_index[c] for c in tweet] for tweet in tweets]

        x = np.asarray([np.asarray([token for token in tweet[:-1]]) for tweet in
            tokenized_tweets])
        y = np.asarray([np.asarray([token for token in tweet[1:]]) for tweet in
            tokenized_tweets])

        return x, y, index_to_char, char_to_index

def get_chars(tweets):
    chars = []

    for c in itertools.chain(*tweets):
        if c not in chars:
            chars.append(c)

    return chars

def shuffle(x, y):
    random_mask = np.random.permutation(np.arange(x.shape[0]))

    return x[random_mask], y[random_mask]
