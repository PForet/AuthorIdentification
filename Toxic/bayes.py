from utils import load_dataframes, _all_labels
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss

sample, test, train = load_dataframes()

Y_dict = {k:train[k] for k in _all_labels}
X = train['comment_text']

y_train, y_val = {}, {}

for k in _all_labels:
    X_train, X_val, y_train[k], y_val[k] = train_test_split(X, Y_dict[k], random_state = 1)
    
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train)
X_train = tf_transformer.transform(X_train)
X_val = tf_transformer.transform(X_val)

    
print("Naive Bayes classifier")
for k in _all_labels:
    gnb =  MultinomialNB()
    gnb.fit(X_train, y_train[k])
    prediction = gnb.predict_proba(X_val)
    print("Naive bayes Logloss for {} : {}".format(k, log_loss(y_val[k], prediction)))
