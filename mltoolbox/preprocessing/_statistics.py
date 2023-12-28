import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.utils import check_array
from ..base import BaseEstimator, FrameTransformerMixin


class StatisticsSelect(BaseEstimator, FrameTransformerMixin):
    def __init__(self, threshold=0.8, *, total_iv=None, percentiles=None, match_list=None):
        self.threshold = threshold
        self.total_iv = total_iv
        self.percentiles = percentiles
        self.match_list = match_list
        self.statistics = None
        self.match_rate = None

    def fit(self, X: DataFrame, y=None, **fit_params):
        if not isinstance (X, pd.DataFrame):
            raise TypeError("Input data must be DataFrame. got type {(} instead.".format(type(X)))
        if self.total_iv:
            self.total_iv = self._check_iv(X)
        self.statistics = self._get_statistic_info(X)
        self.match_rate = self._get_match_rate(X)
        return self
    
    def transform(self, X):
        X = self._ensure_dataframe(X)
        _mask = self.match_rate < self.threshold
        mask = ~_mask
        data = X.iloc[:, mask]
        return data
    
    def statistics_(self):
        statistics_df = self.statistics
        statistics_df['total_iv'] = self.total_iv
        statistics_df['match_rate'] = self.match_rate
        return statistics_df
    
    def _check_iv(self, X):
        total_iv = check_array(self.total_iv, accept_sparse='csr', force_all_finite=True, ensure_2d=False, dtype=None)
        len_iv = len(self.total_iv)
        if len_iv != len(X.columns):
            raise ValueError("total iv has a different length to x.columns.")
        return total_iv
    
    def _get_statistic_info(self, X):
        if self.percentiles:
            if not isinstance(self.percentiles, (list, np.array, pd.Series)):
                raise TypeError("percentiles must be one type of (list, np.array, pd.Series), got type {} instead.".format(type(self.percentiles)))
            statistics_info = X.describe(self.percentiles).T
        else:
            statistics_info = X.describe().T
        return statistics_info
    
    def _get_match_rate(self, X):
        if not self.match_list:
            num = X.notna().sum()
            match_amount = np.array(num)
            match_rate = match_amount / X.shape[0]
        else:
            initial_match_rate = [np.nan] * len(X.columns)
            match_index = [list(X.columns).index(i) for i in self.match_list]
            num = X[self.match_list].notna().sum()
            match_amount = np.array(num)
            match_rate = match_amount / X.shape[0]
            for k, v in zip(match_index, match_rate):
                initial_match_rate[k] =v
            match_rate = np.array(initial_match_rate)
        return match_rate
