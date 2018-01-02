from utils import load_dataframes, _all_labels, remove_non_strings, dict_to_submit
import numpy as np
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB as model
from NBSVM import NBSVM as model
from sklearn.metrics import log_loss

sample, test, train = load_dataframes()

Y_dict = {k:train[k] for k in _all_labels}
X = train['comment_text']

y_train, y_val = {}, {}

for k in _all_labels:
    X_train, X_val, y_train[k], y_val[k] = train_test_split(X, Y_dict[k], random_state = 1, test_size=0.01)
    

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


tf_transformer = TfidfVectorizer(use_idf=True,
                                 #tokenizer=tokenize,
                                 #ngram_range=(1,2),
                                 sublinear_tf=True,
                                 min_df=12,
                                 max_df=0.5).fit(X_train)
"""
tf_transformer = CountVectorizer(min_df=12,
                                 max_df=0.5,
                                 ngram_range=(1,2)).fit(X_train)
"""
X_train = tf_transformer.transform(X_train)
X_val = tf_transformer.transform(X_val)

X_test = tf_transformer.transform(remove_non_strings(test['comment_text']))
predictions_dict = {}

logloss_ = []

print("Naive Bayes classifier")
for k in _all_labels:
    gnb =  model(alpha=0.2,C=2)
    gnb.fit(X_train, y_train[k])
    prediction = [e[1] for e in gnb.predict_proba(X_val)]
    print("Naive bayes Logloss for {} : {}".format(k, log_loss(y_val[k], prediction)))
    logloss_.append(log_loss(y_val[k], prediction))
    predictions_dict[k] = [e[1] for e in gnb.predict_proba(X_test)]

print("Mean logloss : {}".format(np.mean(logloss_)))
dict_to_submit(predictions_dict,"BNBv2_all.csv")