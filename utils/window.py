from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np 

class WindowSequencer(BaseEstimator,TransformerMixin):
    
    def __init__(self,window_size=None,column_to_be_sequenced='cbg',seed=42):
        self.window_size = window_size
        self.column_to_be_sequenced = column_to_be_sequenced
        
    def fit(self, X, y=None):
        self._info(f"Load pandas dataframe of shape {X.shape}")
        return self 
    
    def transform(self,X):
        
        shape = X.shape

        num_windows = X.shape[0] // self.window_size 
        self._info(f"Compute the number of available windows={num_windows} of size {self.window_size}")

        X_trimmed = X.iloc[:num_windows * self.window_size]
        self._info(f"Trim dataframe from {shape} into size {X_trimmed.shape}")
        
        X_reshaped = X_trimmed.values.reshape(num_windows,self.window_size,12)       
        self._info(f"Reshape dataframe to size {np.shape(X_reshaped)} ")
       
        return X_reshaped
    
    def _info(self, msg:str) -> None:
        """Info print utility to know which class prints to terminal."""
        print("\033[36m[WindowSequencer]: ", msg, "\033[0m\n")


class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='constant', fill_value=None):
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns)
