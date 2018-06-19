# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 01:00:31 2017

@author: yanghaocsg@gmail.com
"""
import pandas as pd
import numpy as np
import os
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

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

import pickle
from collections import Counter
def run():
    X_train, X_test, y_train, y_test = pickle.load(open('data.pic', 'rb'))
    print('after load', Counter(y_train), Counter(y_test))

    # MultinomialNB Classifier
    vect = TfidfVectorizer(stop_words='english',
                           token_pattern=r'\b\w{2,}\b',
                           min_df=1, max_df=0.1,
                           ngram_range=(1,2))
    mnb = MultinomialNB(alpha=2)
    svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)
    mnb_pipeline = make_pipeline(vect, mnb)
    svm_pipeline = make_pipeline(vect, svm)
    mnb_cv = cross_val_score(mnb_pipeline, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1)
    svm_cv = cross_val_score(svm_pipeline, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1)
    print('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % mnb_cv.mean())
    # 0.28284
    print('\nSVM Classifier\'s Accuracy: %0.5f\n' % svm_cv.mean())
    # 0.27684

if __name__=='__main__':
    run()