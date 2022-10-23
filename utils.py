import re
import random
import time
from collections import defaultdict
import math
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
STOPWORDS = stopwords.words('english')
# Minimum frequency of a word to be included in the vocabulary
MIN_FREQ = 40


# Clean and tokenize
def prep_sentence(
    sentence, remove_special_chars=True, remove_stopwords=True,
    replace_numbers='num', ignore_case=True
):
    if remove_special_chars:
        # Remove special characters
        sentence = re.sub('[^a-zA-Z1-9]', ' ', sentence)
    if replace_numbers:
        # Replace numbers with a token. Could be useful to limit the vocabulary
        sentence = re.sub('\s\d+[\.*\d+]*', f' {replace_numbers} ', sentence)
    if ignore_case:
        # Case is not important most of the time
        sentence = sentence.lower()
    # Tokenize the sentence into words
    words = sentence.split()
    if remove_stopwords:
        # Stopwords are not always useful
        words = [w.lower() for w in words if w not in STOPWORDS]
    return words


def build_word2idx(
    sentence_file, min_freq=0, num_words=None, return_sentence_words=False
):
    word_freq = defaultdict(int)
    sent_words = []
    with open(sentence_file, 'r') as sf:
        print('>>> Building word frequncy table...')
        start = time.time()
        sent = sf.readline()
        sent_count = 0
        while sent:
            sent_count += 1
            words = prep_sentence(sent.strip())
            if return_sentence_words and words:
                sent_words.append(words)
            for word in words:
                word_freq[word] += 1
            sent = sf.readline()
        print(f'<<< Word frequency table ready! Took {round(time.time() - start, 3)} s')
        print(f'Read {sent_count} sentences.')
    # Sort word frequencies in descending order. Useful for negative sampling
    # later on
    start = time.time()
    word_freq = {w: f for w, f in word_freq.items() if f >= min_freq}
    word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    # We could put frequency and rank in a tuple so that we do not end up with
    # two dicts, but it would be cumbersome
    print('>>> Building word-to-index table...')
    word2idx = {}
    for idx, word in enumerate(list(word_freq.keys())):
        if num_words is not None and idx >= num_words:
            break
        word2idx[word] = idx
    word2idx['oov'] = len(word2idx)
    print(f'<<< {len(word2idx)} + 1 words indexed. Took {round(time.time() - start, 3)} s')
    ret_vals = (word2idx, word_freq)
    if return_sentence_words:
        ret_vals += (sent_words,)
    return ret_vals


def load_datasets(names=[]):
    dfs = tuple()
    for name in names:
        if name == 'train':
            print('>>> Loading labled training dataset...')
            train = pd.read_csv(
                '../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',
                header=0, delimiter='\t', quoting=3
            )
            print(f'<<< Loaded {train.shape[0]} labled training examples\n')
            dfs += (train,)

        elif name == 'unlabled_train':
            print('>>> Loading unlabled training dataset...')
            unlabled_train = pd.read_csv(
                '../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip',
                header=0, delimiter='\t', quoting=3
            )
            print(f'<<< Loaded {unlabled_train.shape[0]} unlabled training examples\n')
            dfs += (unlabled_train,)
        elif name == 'test':
            print('>>> Loading test set...')
            test = pd.read_csv(
                '../input/word2vec-nlp-tutorial/testData.tsv.zip',
                header=0, delimiter='\t', quoting=3
            )
            print(f'<<< Loaded {test.shape[0]} test examples\n')
            dfs += (test,)

    return dfs


def preprocess_manually():
    # This project has two parts, training a skip-gram model to build word
    # embeddings, and using these embeddings to perform sentiment analysis to
    # classify movie reviews to positive and negative.
    # labled_train_df is used in both parts, while unlabled_train_df is used
    # only to build word embeddings.
    labled_train_df, unlabled_train_df = load_datasets([
        'train',
        'unlabled_train'
    ])
    rand_idx = random.randint(0, labled_train_df.shape[0])
    print(f'Sample review: {labled_train_df.iloc[rand_idx]["review"]}')
    print(f'Sentiment: {labled_train_df.iloc[rand_idx]["sentiment"]}')

    all_words = []
    start = time.time()
    labled_train_df['review'] = labled_train_df.review.apply(
        review_to_sentence_words,
        args=(sent_tokenize, all_words, True, False, 'num')
    )
    print(f'Labled reviews tokenized in {time.time() - start} seconds')
    print(f'Sample review Words: {labled_train_df.iloc[rand_idx]["review"]}')

    start = time.time()
    unlabled_train_df['review'] = unlabled_train_df.review.apply(
        review_to_sentence_words,
        args=(sent_tokenize, all_words, True, False, 'num')
    )
    print(f'Unlabled reviews tokenized in {time.time() - start} seconds')

    start = time.time()
    fdist = nltk.FreqDist(word for word in all_words)
    print(f'Frequency distribution built in {time.time() - start} seconds')
    del all_words
    # Since each movie occurs 30 times, we set the minimum word count to 40,
    # to avoid attaching too much importance to individual movie titles
    print(f'>>> Total of {len(fdist)} words counted')
    word_freq = {w: c for w, c in fdist.items() if c >= MIN_FREQ}
    print(f'>>> Selected {len(word_freq)} words appeared {MIN_FREQ} times or more')
    n_common = 20
    print(f'>>> Most common {n_common} words:{fdist.most_common(n_common)}')

    word2idx = {}
    idx2word = {}
    start = time.time()
    for idx, word in enumerate(word_freq.keys()):
        word2idx[word] = idx
        idx2word[idx] = word
    print(f'Word-index and index-words mappings built it {time.time() - start} seconds')
    word2idx['oov'] = len(word2idx)
    idx2word[len(idx2word)] = 'oov'
    del fdist

    start = time.time()
    labled_train_df['review'] = labled_train_df.review.apply(
        review_to_indices,
        args=(word2idx,)
    )
    print(f'Labled review tokens mapped into indices in {time.time() - start} seconds')
    print(f'Sample review tokens: {labled_train_df.iloc[rand_idx]["review"]}')
    start = time.time()
    unlabled_train_df['review'] = unlabled_train_df.review.apply(
        review_to_indices,
        args=(word2idx,)
    )
    print(f'Unlabled review tokens mapped into indices in {time.time() - start} seconds')

    test_df = load_datasets(['test'])
    start = time.time()
    test_df['review'] = test_df.review.apply(
        review_to_sentence_words,
        args=(sent_tokenize, None, True, False, 'num')
    )
    print(f'Test set tokenized in {time.time() - start} seconds')
    start = time.time()
    test_df['review'] = test_df.review.apply(
        review_to_indices,
        args=(word2idx,)
    )
    print(f'Test review tokens mapped into indices in {time.time() - start} seconds')

    sequences = [sent for rev in labled_train_df.review.tolist() for sent in rev]
    sequences += [sent for rev in unlabled_train_df.review.tolist() for sent in rev]

    return sequences


def review_to_sentence_words(
    review, tokenizer, all_words=None, remove_stopwords=False,
    remove_numbers=False, replace_numbers=None
):
    rev_sent_words = []
    for sentence in tokenizer(review.strip()):
        if not sentence:
            continue
        words = sentence_to_words(
            sentence, remove_stopwords, remove_numbers, replace_numbers
        )
        rev_sent_words.append(words)
        if all_words is not None:
            all_words.extend(words)
    return rev_sent_words


def review_to_indices(review, word2idx):
    rev2idx = []
    for sentence in review:
        sent2idx = []
        for word in sentence:
            sent2idx.append(word2idx.get(word, word2idx['oov']))
        rev2idx.append(sent2idx)
    return rev2idx


def sentence_to_words(
    sentence, remove_stopwords=False, remove_numbers=False,
    replace_numbers=None
):
    sentence_text = BeautifulSoup(sentence, features="lxml").get_text()
    sentence_text = re.sub('[^a-zA-Z1-9]', ' ', sentence_text)
    if remove_numbers:
        sentence_text = re.sub('[1-9]', ' ', sentence_text)
    elif replace_numbers:
        sentence_text = re.sub(
            '\s\d+[\.*\d+]*', f' {replace_numbers} ', sentence_text
        )
    words = sentence_text.lower().split()
    if remove_stopwords:
        words = [w.lower() for w in words if w not in STOPWORDS]
    return words


def log_uniform_distribution(vocab_size):
    probs = []
    for rank in range(vocab_size):
        # Equation from https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler
        probs.append((math.log(rank + 2) - math.log(rank + 1)) / math.log(vocab_size + 1))
    return probs
