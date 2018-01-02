from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np

class NBSVM:
    
    def __init__(self, alpha=1, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
    
    def fit(self, X, y):
        f_1 = csr_matrix(y).transpose()
        f_0 = csr_matrix(np.subtract(1,y)).transpose()
        p_ = np.add(self.alpha, X.multiply(f_1).sum(axis=0))
        q_ = np.add(self.alpha, X.multiply(f_0).sum(axis=0))
        p_normed = np.divide(p_, np.sum(p_))
        q_normed = np.divide(q_, np.sum(q_))
        self.r_ = np.log(np.divide(p_normed, q_normed))
        self.f_bar_ = X.multiply(self.r_)
        self.lr_ = LogisticRegression(dual=True, **self.kwargs)
        self.lr_.fit(self.f_bar_, y)
        
    def predict_proba(self, X):
        return self.lr_.predict_proba(X.multiply(self.r_))
        

