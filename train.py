import numpy as np
import argparse
from datetime import datetime as dt
from lstm import LSTM
from data_set_utils import build_training_set, shuffle


START = '<START>'
END = '<END>'

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--learning_rate', type=float, dest='learning_rate',
    default=1e-3)
parser.add_argument('-b', '--batch-size', type=int, dest='batch_size',
    default=1)
parser.add_argument('-d', '--save-directory', type=str, dest='save_dir',
    default='./saved-models/word-lstm.ckpt')
parser.add_argument('-e', '--embedding-size', type=int, dest='embedding_size',
    default=64)
parser.add_argument('-i', '--infile', type=str, dest='infile',
    default='../../data-sets/donald-trump-data/trump-tweets.txt')
parser.add_argument('-l', '--layer-sizes', nargs='+', dest='layer_sizes',
    default=[256, 256, 256])
parser.add_argument('-n', '--n-epochs', type=int, dest='n_epochs', default=20)
parser.add_argument('-s', '--n-tweets', type=int, dest='n_tweets', default=5)
args = parser.parse_args()

np.random.seed(0)

x_train, y_train, index_to_char, char_to_index = build_training_set(args.infile)
model = LSTM(embedding_size=args.embedding_size,
             layer_sizes=args.layer_sizes,
             n_classes=len(index_to_char),
             learning_rate=args.learning_rate,
             batch_size=args.batch_size)

loss = [0] * args.n_epochs
examples_seen = [0] * args.n_epochs
start_time = dt.now()

print("\n")

for epoch in range(args.n_epochs):
    x_train, y_train = shuffle(x_train, y_train)
    for x, y in zip(x_train, y_train):
        loss[epoch] += model.sgd_step(np.asmatrix(x), np.asmatrix(y))
        examples_seen[epoch] += 1
        avg_loss = loss[epoch] / examples_seen[epoch]

        print(f'[AVG ERROR FOR EPOCH {epoch + 1}]: ', end='')
        print('{0:.5f}'.format(avg_loss), end='')
        print(f'\t[EXAMPLES SEEN]: {examples_seen[epoch]} / {x_train.shape[0]}',
            end='')
        print(f'\t[TIME ELAPSED]: {str(dt.now() - start_time)}', end='\r')

        # Generate some tweets after every epoch
        print('\nGenerating tweets...\n')
        for _ in range(args.n_tweets):
            print(''.join(model.generate_sequence(10, 140, START, END, index_to_char,
            char_to_index)[1:-1]), end='\n\n')
        print("\n")
