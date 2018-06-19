import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from collections import Counter
import pickle
def prebuild():
    df = pd.read_csv('./data/ign.csv')
    df = df[df.score_phrase != 'Disaster']
    title = list(df.title)
    label = list(df.score_phrase)

    print('title_counter', Counter(label))
    X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)
    print('X_train', X_train[:3])
    print('Y_train', y_train[:3])
    open('./data.train', 'w+').write('\n'.join(['__label__%s %s' % (l, t) \
                                                for (t, l) in zip(X_train, y_train)]))
    open('./data.test', 'w+').write('\n'.join(['__label__%s %s' % (l, t) \
                                               for (t, l) in zip(X_test, y_test)]))
    pickle.dump((X_train, X_test, y_train, y_test), open('data.pic','wb+'))

    print('before dump', Counter(y_train), Counter(y_test))
    X_train, X_test, y_train, y_test = pickle.load(open('data.pic', 'rb'))
    print('after load', X_train[:3], Counter(y_train), Counter(y_test))


if __name__=='__main__':
    prebuild()

