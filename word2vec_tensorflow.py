import os
import re
import csv
import time
import string
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import tensorflow as tf
from tensorflow.keras import layers
import tqdm

FILE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(FILE_DIR, '../data')


BATCH_SIZE = 1024
VOCAB_SIZE = 32768
SEQ_LEN = 15
WIN_SIZE = 5
EMBED_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 10000
SEED = 42
# The window size. Determines the span of words on either side of a target_word
# that can be considered a context word
# The number of negative samples per a positive context word
NUM_NS = 4


def process():
    # rewrite_reviews_into_sentences()
    sequences = preprocess([os.path.join(INPUT_DIR, 'train_sentences.txt')])

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=WIN_SIZE,
        num_ns=NUM_NS,
        vocab_size=VOCAB_SIZE,
        seed=SEED,
    )
    targets = np.array(targets)
    contexts = np.array(contexts)[:, :, 0]
    labels = np.array(labels)

    # Tuning dataset for performance
    dataset_sg = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset_sg = dataset_sg.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset_sg = dataset_sg.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = Word2Vec(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_SIZE,
        num_ns=NUM_NS
    )
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.fit(dataset_sg, epochs=20)


def rewrite_reviews_to_sentences():
    # Rewrite reviews into separate sentences befoe tokenization
    with open(os.path.join(INPUT_DIR, 'train_sentences.txt'), 'w', newline='') as tdof:
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
            print(f'>>> Finished in {time.time() - start} seconds')


def preprocess(input_paths):
    print(f'>>> Loading text from {input_paths}')
    text_ds = tf.data.TextLineDataset(input_paths).filter(
        lambda x: tf.cast(tf.strings.length(x), bool)
    )

    print('>>> Preprocessing text')
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQ_LEN,
    )
    print('>>> Building vocabulary')
    vectorize_layer.adapt(text_ds.batch(BATCH_SIZE))

    text_vector_ds = text_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

    sequences = list(text_vector_ds.as_numpy_iterator())

    return sequences


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(
        lowercase,
        '[%s]' % re.escape(string.punctuation),
        ''
    )


# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    print('>>> Generating training data')
    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    print(f'>>> Sampling table size {len(sampling_table)}')
    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0,
        )
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype='int64'), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name='negative_sampling',
            )
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype='int64')
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    return targets, contexts, labels


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name='w2v_embedding',
        )
        self.context_embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns+1,
        )

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots


if __name__ == '__main__':
    process()
