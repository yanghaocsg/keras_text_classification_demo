
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


# Hierarchical Model with Attention
class AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)  # (x, 40, 1)
        uit = K.squeeze(uit, -1)  # (x, 40)
        uit = uit + self.b  # (x, 40) + (40,)
        uit = K.tanh(uit)  # (x, 40)

        ait = uit * self.u  # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait)  # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())  # (x, 40)
            ait = mask * ait  # (x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])




# 0.4487
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


        # one-hot mlp
        x_train = tokenizer.sequences_to_matrix(x_train_word_ids, mode='binary')
        x_test = tokenizer.sequences_to_matrix(x_test_word_ids, mode='binary')

        inputs = Input(shape=(20,), dtype='float64')
        embed = Embedding(len(vocab) + 1, 300, input_length=20)(inputs)
        gru = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embed)
        attention = AttLayer()(gru)
        output = Dense(num_labels, activation='softmax')(attention)
        model = Model(inputs, output)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('model summary\n', model.summary())

        model.fit(x_train_padded_seqs, y_train,
                  batch_size=128,
                  epochs=6,
                  validation_data=(x_test_padded_seqs, y_test))


kerasMlp = KerasMlp()
if __name__=='__main__':
    kerasMlp.run()