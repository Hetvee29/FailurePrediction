from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np

class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.type_encoder = OrdinalEncoder(categories=[['L', 'M', 'H']])
        self.failure_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.scale_cols = ['Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min',
                           'Air_temperature_C', 'Process_temperature_C']

    def fit(self, X, y=None):
        self.type_encoder.fit(X[['Type']])
        if 'Type_of_failure' in X.columns:
            self.failure_encoder.fit(X['Type_of_failure'])
        self.scale_cols = [col for col in self.scale_cols if col in X.columns]
        self.scaler.fit(X[self.scale_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X['Type'] = self.type_encoder.transform(X[['Type']])
        if 'Type_of_failure' in X.columns:
            X['Type_of_failure'] = self.failure_encoder.transform(X['Type_of_failure'])
        X[self.scale_cols] = self.scaler.transform(X[self.scale_cols])
        return X
