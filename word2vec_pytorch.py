import os
import time
import csv
import pandas
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset

FILE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(FILE_DIR, '../input/word2vec-nlp-tutorial')
BBUFF_SIZE = 1024 * 1024


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


class SentenceDataset(Dataset):
    def __init__(self, sentence_file):
        self.sentence_file = sentence_file

    def __len__(self):
        # Buffering byte arrays to be used laters to count endline characters.
        def _buffer_generator(reader):
            b = reader(BBUFF_SIZE)
            while b:
                yield b
                b = reader(BBUFF_SIZE)

        with open(self.sentence_file, 'r') as sfp:
            buf_generator = _buffer_generator(sfp.raw.read)
            return sum(buf.count(b'\n') for buf in buf_generator)

    def __get_item__(self, idx):
        pass


def load_datasets():
    pass


def process():
    # rewrite_reviews_to_sentences()
    pass


if __name__ == '__main__':
    process()
