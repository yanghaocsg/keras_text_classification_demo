
from keras import backend
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import  BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers
from keras.engine.topology import Layer

import numpy as np
from sklearn import preprocessing

import pickle, os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class KerasMlp():
    def __init__(self):
        pass

    def run(self):
        #load data
        X_train, X_test, y_train, y_test = pickle.load(open('data.pic', 'rb'))
        print(X_train[:3])
        print('after load', X_train[:3], Counter(y_train), Counter(y_test), 'data type y_train', type(y_train))

        y_labels = list(y_train + y_test)
        le = preprocessing.LabelEncoder()
        le.fit(y_labels)
        num_labels = len(y_labels)
        y_train = to_categorical([le.transform([x])[0] for x in  y_train], num_labels)
        y_test = to_categorical([le.transform([x])[0] for x in   y_test], num_labels)

        # load glove word embedding data
        GLOVE_DIR = "/root/research/data"
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding = 'utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('embedding size', len(embeddings_index))

        # take tokens and build word-id dictionary
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' ,lower=True ,split=" ")
        tokenizer.fit_on_texts(X_train + X_test)
        vocab = tokenizer.word_index

        # Match the word vector for each word in the data set from Glove
        '''
        embedding_matrix = np.zeros((len(vocab) + 1, 300))
        for word, i in vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        '''

        # Match the input format of the model
        x_train_word_ids = tokenizer.texts_to_sequences(X_train)
        x_test_word_ids = tokenizer.texts_to_sequences(X_test)
        x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=20)
        x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=20)

        # CNN model
        model = Sequential()
        model.add(Embedding(len(vocab) + 1, 256, input_length=20))

        # Convolutional model (3x conv, flatten, 2x dense)
        model.add(Convolution1D(256, 3, padding='same'))
        model.add(MaxPool1D(3, 3, padding='same'))
        model.add(Convolution1D(128, 3, padding='same'))
        model.add(MaxPool1D(3, 3, padding='same'))
        model.add(Convolution1D(64, 3, padding='same'))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(num_labels, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print('model summary\n', model.summary())

        model.fit(x_train_padded_seqs, y_train,
                  batch_size=32,
                  epochs=12,
                  validation_data=(x_test_padded_seqs, y_test))


kerasMlp = KerasMlp()
if __name__=='__main__':
    kerasMlp.run()