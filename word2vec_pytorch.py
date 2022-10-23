import os
import csv
import time
import math
import pickle
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import utils
import common

# Data
FILE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(FILE_DIR, '../data')
SENTENCE_FILE = os.path.join(INPUT_DIR, 'movie_review_sentences.txt')
DATA_FILE = os.path.join(INPUT_DIR, 'movie_review_word2vec_samples.pkl')
MODEL_FILE = os.path.join(FILE_DIR, 'models/Word2VecNS.pth')

# Parameters
EPOCHS = 3
BATCH_SIZE = 64
VOCAB_SIZE = 4096
WINDOW_SIZE = 2
NUM_NS = 4
EMBED_SIZE = 300


def rewrite_reviews_to_sentences(sentence_file):
    # Rewrite reviews into separate sentences befoe tokenization
    with open(sentence_file, 'w', newline='') as tdof:
        with open(os.path.join(INPUT_DIR, 'labeledTrainData.tsv'), newline='') as ltdif:
            review_reader = csv.reader(ltdif, delimiter='\t', quotechar='"')
            print('>>> Rewriting labled reviews to clean sentences...')
            start = time.time()
            for review in review_reader:
                for sentence in sent_tokenize(review[2].strip()):
                    if not sentence:
                        continue
                    sentence_clean = BeautifulSoup(sentence).get_text()
                    tdof.write(f'{sentence_clean}\n')
            print(f'>>> Finished in {time.time() - start} seconds')
        with open(os.path.join(INPUT_DIR, 'unlabeledTrainData.tsv'), newline='') as ultdif:
            review_reader = csv.reader(ultdif, delimiter='\t', quotechar='"')
            print('>>> Rewriting unlabled reviews to clean sentences...')
            start = time.time()
            for review in review_reader:
                for sentence in sent_tokenize(review[1].strip()):
                    if not sentence:
                        continue
                    sentence_clean = BeautifulSoup(sentence).get_text()
                    tdof.write(f'{sentence_clean}\n')
            print(f'<<< Finished in {time.time() - start} seconds')


def generate_samples(sentences, window_size, num_ns, word2idx, vocab_size):
    print('>>> Generating Word2Vec samples...')
    start = time.time()
    targets = []
    contexts = []
    labels = []
    sample_dist = utils.log_uniform_distribution(vocab_size)
    for sentence in sentences:
        for i in range(len(sentence)):
            # We use word indices to train the model not the words themselves.
            # That's what Embedding layers take too
            center_word = word2idx.get(sentence[i], word2idx['oov'])
            # Context words before the center word
            context_words = sentence[max(0, i - window_size):i]
            # Context words after the center word
            context_words += sentence[min(len(sentence) - 1, i + 1):min(len(sentence) - 1, i + window_size + 1)]
            context_words = list(map(lambda w: word2idx.get(w, word2idx['oov']), context_words))
            for cw in context_words:
                targets.append(center_word)
                ns_context = [cw]
                neg_samp_cand = np.random.choice(range(vocab_size), size=num_ns + len(context_words) + 1, replace=False, p=sample_dist).tolist()
                negative_samples = []
                for neg_samp in neg_samp_cand:
                    if len(negative_samples) < num_ns and neg_samp not in [center_word] + context_words:
                        negative_samples.append(neg_samp)
                ns_context.extend(negative_samples)
                contexts.append(ns_context)
                labels.append([1] + [0] * num_ns)
                if len(targets) % 100000 == 0:
                    print(f'... {len(targets)} samples')
    print(f'<<< {len(targets)} samples created! Took {round(time.time() - start, 3)} seconds')
    return targets, contexts, labels


class Word2VecNsDs(Dataset):
    def __init__(self, targets, contexts, labels):
        self.targets = torch.tensor(targets)
        self.contexts = torch.tensor(contexts)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.targets[idx], self.contexts[idx]), self.labels[idx]


def load_batch_to_device(xb, yb):
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return [x.to(dev) for x in xb], yb.to(dev)


# TODO revisit to see if we need to add more layers. Also, looping over the
# batch is very inefficient, try to find a way to vectorize the operation.
class Word2VecNS(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(vocab_size, embed_size) / math.sqrt(vocab_size))

    def forward(self, xb):
        yhb = None
        for target, context in zip(*xb):
            yh = self.embed[target] @ self.embed[context].T
            if yhb is None:
                yhb = yh.unsqueeze(dim=0)
            else:
                yhb = torch.cat((yhb, yh.unsqueeze(dim=0)), dim=0)
        return yhb


def process(
    sentence_file, window_size, num_ns, batch_size, epochs, vocab_size,
    embed_size, data_file_in=None, data_file_out=None, model_file_in=None,
    model_file_out=None,
):
    # rewrite_reviews_to_sentences(sentence_file=SENTENCE_FILE)
    # Generating samples takes LOOONG! It makes sense to save them for
    # later runs
    if data_file_in:
        print(f'>>> Loading data from {data_file_in}...')
        with open(data_file_in, 'rb') as sfi:
            start = time.time()
            word2idx, word_freq, sentences, targets, contexts, labels = pickle.load(sfi)
            print('<<< Loaded:')
            print(f'   {len(sentences)} sentences')
            print(f'   {len(word2idx)} word indices')
            print(f'   {len(targets)} samples')
            print(f'Took {round(time.time() - start, 3)} seconds')
    else:
        word2idx, word_freq, sentences = utils.build_word2idx(
            sentence_file=SENTENCE_FILE,
            num_words=VOCAB_SIZE,
            return_sentence_words=True,
        )
        targets, contexts, labels = generate_samples(
            sentences[:10000], window_size, num_ns, word2idx, vocab_size
        )
        if data_file_out:
            with open(data_file_out, 'wb') as sfo:
                pickle.dump((word2idx, word_freq, sentences, targets, contexts, labels), sfo)
    print('\n>>> Most common words:')
    count = 0
    for word, freq in word_freq.items():
        if count > 20:
            break
        print(f'{word}: {freq}')
        count += 1
    print('\n>>> Sample sentences:')
    for sent in sentences[:5]:
        print(f'  {sent}')
    print()

    print(f'Sample: targe={targets[0]}, context={contexts[0]}, label={labels[0]}\n\n')

    train_ds = Word2VecNsDs(targets, contexts, labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_dl = common.WrappedDataLoader(train_dl, load_batch_to_device)

    # The +1 is to count for oov
    if model_file_in:
        model = torch.load(model_file_in)
    else:
        model = Word2VecNS(vocab_size + 1, embed_size)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)

    common.fit(
        epochs=epochs,
        model=model,
        loss_func=F.cross_entropy,
        opt=optim.Adam(model.parameters(), lr=0.001),
        train_dl=train_dl,
        valid_dl=None,
    )

    if model_file_out:
        torch.save(model, model_file_out)


if __name__ == '__main__':
    process(
        sentence_file=SENTENCE_FILE,
        window_size=WINDOW_SIZE,
        num_ns=NUM_NS,
        epochs=EPOCHS,
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        batch_size=BATCH_SIZE,
        # data_file_out=DATA_FILE,
        data_file_in=DATA_FILE,
        # model_file_in=MODEL_FILE,
        model_file_out=MODEL_FILE,
    )
