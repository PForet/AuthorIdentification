from utils import load_dataframes, _all_labels, remove_non_strings, dict_to_submit
import numpy as np
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB as model
from sklearn.ensemble import GradientBoostingClassifier
from NBSVM import NBSVM as model
from sklearn.metrics import log_loss

sample, test, train = load_dataframes()

Y_dict = {k:train[k] for k in _all_labels}
X = train['comment_text']

y_train, y_val = {}, {}

for k in _all_labels:
    X_train_r, X_val_r, y_train[k], y_val[k] = train_test_split(X, Y_dict[k], random_state = 1)
    

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
#def tokenize(s): return re_tok.sub(r' \1 ', s).split()
def tokenize(s): return re.findall(r"[\w']+|!|@|!!",s)

def processing(s):
    re_links = re.compile("http.*?\s")
    s = re_links.sub("_link_", s)
    return s.lower()

tf_transformer = TfidfVectorizer(use_idf=True,
                                 tokenizer=tokenize,
                                 #ngram_range=(1,2),
                                 sublinear_tf=True,
                                 preprocessor=processing,
                                 min_df=10,
                                 max_df=0.9).fit(X_train_r)
"""
tf_transformer = CountVectorizer(min_df=12,
                                 max_df=0.5,
                                 ngram_range=(1,2)).fit(X_train)
"""
X_train = tf_transformer.transform(X_train_r)
X_val = tf_transformer.transform(X_val_r)

X_test = tf_transformer.transform(remove_non_strings(test['comment_text']))
predictions_dict, val_dict, train_dict = {}, {}, {}

logloss_ = []

print("Naive Bayes classifier")
for k in _all_labels:
    gnb =  model(alpha=0.3, C=1.5)
    gnb.fit(X_train, y_train[k])
    prediction = [e[1] for e in gnb.predict_proba(X_val)]
    train_prediction = [e[1] for e in gnb.predict_proba(X_train)]
    print("Naive bayes Logloss for {} : {} with train loss of {}".format(k, 
              log_loss(y_val[k], prediction),log_loss(y_train[k], train_prediction)))
    val_dict[k] = prediction
    train_dict[k] = train_prediction
    logloss_.append(log_loss(y_val[k], prediction))
    predictions_dict[k] = [e[1] for e in gnb.predict_proba(X_test)]
print("Mean logloss : {}".format(np.mean(logloss_)))

print("Assembling")

def dict_to_X(d):
    return np.array([d[k] for k in _all_labels]).transpose()
X_train_pred = dict_to_X(train_dict)
X_val_pred = dict_to_X(val_dict)
X_test_pred = dict_to_X(predictions_dict)


assembled_predictions_dict, assembled_val_dict, assembled_train_dict = {}, {}, {}
assembled_logloss_ = []

for k in _all_labels:
    gnb =  GradientBoostingClassifier(n_estimators=1000)
    gnb.fit(X_train_pred, y_train[k])
    prediction = [e[1] for e in gnb.predict_proba(X_val_pred)]
    train_prediction = [e[1] for e in gnb.predict_proba(X_train_pred)]
    print("Assembled Logloss for {} : {} with train loss of {}".format(k, 
              log_loss(y_val[k], prediction),log_loss(y_train[k], train_prediction)))
    assembled_val_dict[k] = prediction
    assembled_train_dict[k] = train_prediction
    assembled_logloss_.append(log_loss(y_val[k], prediction))
    assembled_predictions_dict[k] = [e[1] for e in gnb.predict_proba(X_test_pred)]
    
    

print("Mean logloss : {}".format(np.mean(assembled_logloss_)))
dict_to_submit(assembled_predictions_dict,"assembled_BNBv2_all.csv")