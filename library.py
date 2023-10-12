import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


#This class will rename one or more columns.
class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
  def __init__(self, renaming_dict: dict):
    assert isinstance(renaming_dict, dict), f'{self.__class__.__name__} constructor expected a dictionary but got {type(renaming_dict)} instead.'
    self.renaming_dict = renaming_dict

  #define fit to do nothing but give warning
  def fit(self, X, y=None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  #write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead.'

    columns_not_found = set(self.renaming_dict.keys()) - set(X.columns)
    if columns_not_found:
        raise AssertionError(f"{self.__class__.__name__}.transform columns not found: {columns_not_found}")

    X_ = X.copy()
    column_mapping = {old_col: new_col for old_col, new_col in self.renaming_dict.items()}
    X_.rename(columns=self.renaming_dict, inplace=True)

    return X_

  #write fit_transform that skips fit
  def fit_transform(self, X, y=None):
    result = self.transform(X)
    return result

class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  #fill in the rest below
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead.'
    assert self.target_column in X.columns, f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'
    X_encoded = pd.get_dummies(X, columns=[self.target_column],
                               dummy_na=self.dummy_na,
                               drop_first=self.drop_first
                               )

    return X_encoded

  def fit_transform(self, X, y=None):
    return self.transform(X)


