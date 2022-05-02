# %% 
# Python modules. 
import numpy as np 
import scipy 
from sklearn.base import BaseEstimator, TransformerMixin 



# %%
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, var_proc): 
        self.var_proc = var_proc 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.var_proc] 



# %%
class ToSparse(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X): 
        return scipy.sparse.csr_matrix(X.values) 



# %%
class ExtractSentiment(BaseEstimator, TransformerMixin):
    def __init__(self, est_pipe, var_proc, var_name="sentiment"): 
        self.est_pipe = est_pipe 
        self.var_proc = var_proc 
        self.var_name = var_name 

    def fit(self, X, y=None):
        return self

    def transform(self, X): 
        # Extract sentiment. 
        X[self.var_name] = X[self.var_proc].apply(self._find_max) 
        X = X.drop(columns=[self.var_proc]) 
        return X 

    def _find_max(self, row): 
        dic_score = self.est_pipe(row).cats 
        sentiment = max(dic_score, key=dic_score.get) 
        return sentiment 



# %%
class ExtractTopic(BaseEstimator, TransformerMixin):
    def __init__(self, est_pipe, var_proc, var_name="theme"): 
        self.est_pipe = est_pipe 
        self.var_proc = var_proc 
        self.var_name = var_name 

    def fit(self, X, y=None):
        return self

    def transform(self, X): 
        # Extract topic. 
        X[self.var_proc] = X[self.var_proc].astype("object") 
        X[self.var_name] = np.argmax(self.est_pipe.transform(X[self.var_proc]), axis=1) 
        X[self.var_name] = "topic_" + X[self.var_name].astype(str) 
        X = X.drop(columns=[self.var_proc]) 
        return X 
