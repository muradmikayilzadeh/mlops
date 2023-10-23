import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce


class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

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


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.fitted = False

  def fit(self, df):
    assert isinstance(df, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
    assert self.target_column in df.columns, f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in df[self.target_column].to_list()])

    sigma = df[self.target_column].std()
    mean = df[self.target_column].mean()
    self.sigma_low = mean - 3 * sigma
    self.sigma_high = mean + 3 * sigma
    self.fitted = True

  def transform(self, df):
    assert self.fitted, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    return self.fit_transform(df)

  def fit_transform(self, df):
      self.fit(df)
      self.df = df.copy()
      self.df[self.target_column] = self.df[self.target_column].clip(lower=self.sigma_low, upper=self.sigma_high)
      self.df.reset_index(drop=True, inplace=True)
      
      return self.df

class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
      assert fence in ['inner', 'outer']
      self.target_column = target_column
      self.fence = fence
      self.fitted = False

  def fit(self,df):
    assert isinstance(df, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
    assert self.target_column in df.columns, f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in df[self.target_column].to_list()])

    q1 = df[self.target_column].quantile(0.25)
    q3 = df[self.target_column].quantile(0.75)

    iqr = q3 - q1

    if self.fence == 'inner':
        self.low = q1 - 1.5 * iqr
        self.high = q3 + 1.5 * iqr
    else:
        self.low = q1 - 3 * iqr
        self.high = q3 + 3 * iqr

    self.fitted = True

  def transform(self, df):
    assert self.fitted, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    return self.fit_transform(df)

  def fit_transform(self, df, x=None):
    self.fit(df)
    self.df = df.copy()
    self.df[self.target_column] = self.df[self.target_column].clip(lower=self.low, upper=self.high)
    self.df.reset_index(drop=True)
    return self.df

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    self.column = column

  def fit (self,df, y=None):
    self.df = df.copy()
    self.iqr = self.df[self.column].quantile(.75) - self.df[self.column].quantile(.25)
    self.med = self.df[self.column].median()
    return self

  def transform(self,df):
    self.fit(df)
    self.df[self.column] -= self.med
    self.df[self.column] /= self.iqr
    self.df[self.column].min(), self.df[self.column].max()
    return self.df
