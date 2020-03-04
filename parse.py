from functions import *

import csv
import re
import math
from math import sqrt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import category_encoders as ce

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.sparse import hstack, coo_matrix

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag


df_train= pd.read_csv('../input/nlp-getting-started/train.csv')
df_test=pd.read_csv('../input/nlp-getting-started/test.csv')
df_train_all = df_train.append(df_test, sort=False)

num_classes = 2
max_words = max_words = 701

add_keyword = True 
parse_pca = False
parse_ce = True

(X, Y, X_test) = parse_train(df_train, df_train_all, df_test, 'll', max_words, add_keyword, parse_pca, parse_ce)

x_train = X
y_train = Y
x_test = X_test

print("Shapes for in : ")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

clf = RandomForestClassifier(max_depth=1000, n_estimators=5000, n_jobs=4, verbose=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print_into_file(y_pred, df_test, 'RF')
