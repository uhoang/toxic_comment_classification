import os 
import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def split_data(data, text, target, 
              idt='id', test_size=.3, 
              output_path='tmp'):
  '''Split data into train and validation sets stratitifed by
  a target distribution'''
  splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
  if not os.path.exists(output_path): os.makedirs(output_path)

  print('Splitting data for %s' % target)
  y = data[target]
  for train_idx, valid_idx in splitter.split(y, y):
    train = data[[idt, text, target]].iloc[train_idx]
    valid = data[[idt, text, target]].iloc[valid_idx]

    ftrain = os.path.join(output_path, 'train_' + target + '.csv')
    fvalid = os.path.join(output_path, 'valid_' + target + '.csv')

    train.to_csv(ftrain, index=False, encoding='utf-8')
    valid.to_csv(fvalid, index=False, encoding='utf-8')


def _preprocess_train_text(text, max_vocab_size=20000, 
                          max_sentence_len=100):
  '''Build training data matrix that map words to indices 
  given a pretrained embedding word vectors'''
  if not isinstance(text, pd.Series):
    raise Exception('text is not a pd.Series object.')
  # fill missing values with '_na_'
  text = text.fillna('_na_').values
  list_sentences = list(text)

  #only top max_vocab_size most frequent words will be counted
  tokenizer = Tokenizer(num_words=max_vocab_size)
  tokenizer.fit_on_texts(list_sentences)

  # convert each text into a sequence of integers
  list_tokens = tokenizer.texts_to_sequences(list_sentences)

  # pad sequences to the same length
  X = pad_sequences(list_tokens, maxlen=max_sentence_len)
  return X, tokenizer 

def preprocess_text(train_text, test_text=None, 
                    max_vocab_size=20000, 
                    max_sentence_len=100):
  '''Build train and test matricies that map words to indices 
  given a pretrained embedding word vectors'''
  X_train, tokenizer = _preprocess_train_text(train_text, max_vocab_size,
                                              max_sentence_len)
  if test_text is not None:
    text = test_text.fillna('_na_').values
    list_tokens = tokenizer.texts_to_sequences(list(text))
    X_test = pad_sequences(list_tokens, maxlen=max_sentence_len)
  else:
    X_test = None
  
  return X_train, X_test, tokenizer.word_index

def _get_embedding_matrix(vocab,
                          max_vocab_size,
                          pretrained_word2vec='input/glove.6B.50d.txt'):
  
  # read the GloVe word vectors (space delimited strings) into a dictionary
  def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
  embedding_index = dict(get_coefs(*o.strip().split()) for o in open(pretrained_word2vec))
  word_vec_size = len(embedding_index[list(embedding_index.keys())[0]])

  # create embedding matrix, with random initialization for words that are not in GloVe
  all_embs = np.stack(embedding_index.values())
  emb_mean, emb_std = all_embs.mean(), all_embs.std()
  max_vocab_size = min(max_vocab_size, len(vocab))
  embedding_matrix = np.random.normal(emb_mean, emb_std, 
                                    (max_vocab_size, word_vec_size))

  # if a word appears in the vocabulary dict and the pretrained word2vec matrix, 
  # replace its randomized values by those in the pretrained matrix 
  for word, i in vocab.items():
    if i >= max_vocab_size: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: 
      embedding_matrix[i] = embedding_vector

  return embedding_matrix

if __name__='__main__':
  df = pd.read_csv('input/train.csv')
  # filter column names starts with id and comment
  targets = df.filter(regex='^(?!.*?id$|comment).*').columns.values
  
  input_path = 'tmp/data/'

  # split data for list of targets [toxic, severe_toxic, ...]
  for target in targets:
    f = os.path.join(input_path, 'train_' + target + '.csv')
    if not os.path.isfile(f):
      split_data(df, text='comment_text', target=target, 
                idt='id', output_path=input_path)
  target = targets[0]
  files = [os.path.join(input_path, e + '_' + targets[0] + '.csv') for e in ['train', 'valid']]
  [train, test] = [pd.read_csv(f) for f in files]
  y_train = train[target]
  y_test = test[target]
  X_train, X_test, vocab = preprocess_text(train_text=train['comment_text'], 
                                          test_text=test['comment_text'],
                                          max_vocab_size=20000,
                                          max_sentence_len=100)
  
  embedding_matrix = _get_embedding_matrix(vocab, max_vocab_size=20000)
