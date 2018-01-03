import numpy as np
from utils import load_dataframes, _all_labels, remove_non_strings
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint 


def bayes_transform(X,y, alpha=1):
    f_1 = csr_matrix(y).transpose()
    f_0 = csr_matrix(np.subtract(1,y)).transpose()
    p_ = np.add(alpha, X.multiply(f_1).sum(axis=0))
    q_ = np.add(alpha, X.multiply(f_0).sum(axis=0))
    p_normed = np.divide(p_, np.sum(p_))
    q_normed = np.divide(q_, np.sum(q_))
    r_ = np.log(np.divide(p_normed, q_normed))
    f_bar_ = X.multiply(r_)
    return f_bar_, r_

sample, test, train = load_dataframes()

Y_dict = {k:train[k] for k in _all_labels}
X = train['comment_text']

y_train, y_val = {}, {}

for k in _all_labels:
    X_train, X_val, y_train[k], y_val[k] = train_test_split(X, Y_dict[k], random_state = 1, test_size=0.1)
    

tf_transformer = TfidfVectorizer(use_idf=True,
                                 #tokenizer=tokenize,
                                 #ngram_range=(1,2),
                                 sublinear_tf=True,
                                 min_df=12,
                                 max_df=0.5).fit(X_train)

X_train = tf_transformer.transform(X_train)
X_val = tf_transformer.transform(X_val)

X_test = tf_transformer.transform(remove_non_strings(test['comment_text']))

X_train, r_ = bayes_transform(X_train, y_train['toxic'])
X_train = X_train.toarray()
X_val = X_val.multiply(r_).toarray()
#########################################


model = Sequential()

model.add(Dense(1024, activation='softmax', input_dim=X_train.shape[1]))
model.add(Dense(2, activation='softmax'))

model.summary()
opt = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='weights.1dense.hdf5', 
                               verbose=1, save_best_only=True)
# Fit the model, using 'verbose'=1 if we specified 'verbose=True' when calling the function (0 else)
model.fit(X_train, [[e,1-e] for e in y_train['toxic']], validation_data=(X_val, [[e,1-e] for e in y_val['toxic']]), callbacks=[checkpointer],
         batch_size=2048, epochs=3)
    
prediction = [e[0] for e in model.predict(X_val)]
print("Naive bayes Logloss for {} : {}".format(k, log_loss(y_val[k], prediction)))
