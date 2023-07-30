from config import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_stories():
    df = pd.read_csv('./data/ROCStories_winter2017 - ROCStories_winter2017.csv')
    df['story'] = df[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].agg(' '.join, axis=1)
    return df['story'].tolist()


def prepare_data(max_df=0.2, min_df=0.001, ids=False):
    texts = load_stories()
    flat_DRM = [item for sublist in DRM_lists for item in sublist]
    texts = [i for i in texts if check_for_words(i, flat_DRM)==True]
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(texts)

    texts = [' '.join(list(vectorizer.inverse_transform(item)[0])) for ind, item in enumerate(X)]
    if ids is True:
        texts.extend([f'id_{n}' for n in range(20)])
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    x_train = X.toarray()
    return x_train, vectorizer


def check_for_words(txt, word_list):
    for w in word_list:
        if w.lower() in txt.lower():
            return True
    return False
