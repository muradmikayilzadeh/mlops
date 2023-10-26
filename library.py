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
from sklearn.neighbors import KNeighborsClassifier 


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

    def fit(self, df, y=None):
        assert isinstance(df, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected DataFrame but got {type(df)} instead.'
        assert self.column in df.columns.to_list(), f'{self.__class__.__name__}.fit unrecognizable column {self.column}.'

        self.iqr = float(df[self.column].quantile(.75) - df[self.column].quantile(.25))
        self.med = df[self.column].median()
        return self

    def transform(self, df):
        assert isinstance(df, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(df)} instead.'
        assert hasattr(self, 'iqr') and hasattr(self, 'med'), f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
        assert self.column in df.columns.to_list(), f'{self.__class__.__name__}.transform unrecognizable column {self.column}.'

        self.df = df.copy()
        self.df[self.column] -= self.med
        self.df[self.column] /= self.iqr
        return self.df

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)  #instantiate with k=5.
  var = []  #collect test_error/train_error where error based on F1 score

  #2 minutes
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)           #predict against training set
      test_pred = model.predict(test_X)             #predict against test set
      train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
      test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
      f1_ratio = test_f1/train_f1          #take the ratio
      var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value
  rs_value  #0.8501547035464532
  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx

titanic_variance_based_split = 107
customer_variance_based_split = 113

titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', ce.TargetEncoder(cols=['Joined'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),  #from chapter 5
    ('scale_fare', CustomRobustTransformer('Fare')),  #from chapter 5
    ('imputer', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False))  #from chapter 6
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', ce.TargetEncoder(cols=['ISP'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer('Age')), #from 5
    ('scale_time spent', CustomRobustTransformer('Time Spent')), #from 5
    ('impute', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)),
    ], verbose=True)
