# %% 
# Python modules. 
import numpy as np 
import scipy 
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.utils.validation import check_is_fitted 
from feature_engine.variable_manipulation import _check_input_parameter_variables 



# %%
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, var_proc): 
        self.var_proc = var_proc 
        self.feature_names_in_ = [] 
        self.feature_names_out = [] 

    def fit(self, X, y=None):
        # Track the input columns or features. 
        self.feature_names_in_ = X.columns.to_list() 
        return self 

    def transform(self, X): 
        X = X[self.var_proc] 
        # Track the output columns or features. 
        self.feature_names_out = X.columns.to_list() 
        return X 

    def get_feature_names_out(self) -> list: 
        # Check if the transformer has fitted or not before user can extract the output features. 
        check_is_fitted(self) 

        if self.feature_names_out: 
            return self.feature_names_out 

        print("No output features to display yet before transforming the features.") 



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
        self.feature_names_in_ = [] 
        self.feature_names_out = [] 

    def fit(self, X, y=None):
        # Track the input columns or features. 
        self.feature_names_in_ = X.columns.to_list() 
        return self

    def transform(self, X): 
        # Extract sentiment. 
        X[self.var_name] = X[self.var_proc].apply(self._find_max) 
        X = X.drop(columns=[self.var_proc]) 

        # Track the output columns or features. 
        self.feature_names_out = X.columns.to_list() 
        return X 

    def _find_max(self, row): 
        dic_score = self.est_pipe(row).cats 
        sentiment = max(dic_score, key=dic_score.get) 
        return sentiment 

    def get_feature_names_out(self) -> list: 
        # Check if the transformer has fitted or not before user can extract the output features. 
        check_is_fitted(self) 

        if self.feature_names_out: 
            return self.feature_names_out 

        print("No output features to display yet before transforming the features.") 



# %%
class ExtractTopic(BaseEstimator, TransformerMixin):
    def __init__(self, est_pipe, var_proc, var_name="theme"): 
        self.est_pipe = est_pipe 
        self.var_proc = var_proc 
        self.var_name = var_name 
        self.feature_names_in_ = [] 
        self.feature_names_out = [] 

    def fit(self, X, y=None): 
        # Track the input columns or features. 
        self.feature_names_in_ = X.columns.to_list() 
        return self

    def transform(self, X): 
        # Extract topic. 
        X[self.var_proc] = X[self.var_proc].astype("object") 
        X[self.var_name] = np.argmax(self.est_pipe.transform(X[self.var_proc]), axis=1) 
        X[self.var_name] = "topic_" + X[self.var_name].astype(str) 

        # Remove the original column or feature. We only keep the processed feature. 
        X = X.drop(columns=[self.var_proc]) 

        # Track the output columns or features. 
        self.feature_names_out = X.columns.to_list() 
        return X 

    def get_feature_names_out(self) -> list: 
        # Check if the transformer has fitted or not before user can extract the output features. 
        check_is_fitted(self) 

        if self.feature_names_out: 
            return self.feature_names_out 

        print("No output features to display yet before transforming the features.") 
