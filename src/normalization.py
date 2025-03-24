import numpy as np
import pandas as pd

from scipy.stats import zscore

'''
Time Series Data Module for managing training with PytorchLightning

Data Format:

(N,C,T)

N: Number of samples
C: Number of channels (univariate = 1, multivariate > 1)
T: Number of timepoints

'''

class ZNormalizer():
    def __init__(self,mean=0,std=1):
        self.mean = mean
        self.std = std

    def transform(self,X):

        self.mean = np.mean(X,axis=2, keepdims=True)
        self.std = np.std(X,axis=2, keepdims=True)
        self.std[np.abs(self.std) < 1e-10] = 1

        z = (X - self.mean) / self.std
        # z = np.nan_to_num(z)
        return z
        
    def fit(self,X):
        self.mean = np.mean(X,axis=2, keepdims=True)
        self.std = np.std(X,axis=2, keepdims=True)
        self.std[np.abs(self.std) < 1e-10] = 1

        return self
    
    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        z = np.nan_to_num(z)
        return z 
    
# Global Z-score
class GlobalZNormalizer():
    def __init__(self,mean=0,std=1):
        self.mean = mean
        self.std = std

    def transform(self,X):

        self.mean = np.mean(X,axis=(1,2), keepdims=True)
        self.std = np.std(X,axis=(1,2), keepdims=True)
        self.std[np.abs(self.std) < 1e-10] = 1

        z = (X - self.mean) / self.std
        # z = np.nan_to_num(z)
        return z
        
    def fit(self,X):
        self.mean = np.mean(X,axis=(1,2), keepdims=True)
        self.std = np.std(X,axis=(1,2), keepdims=True)
        self.std[np.abs(self.std) < 1e-10] = 1

        return self
    
    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        z = np.nan_to_num(z)
        return z 
    

class MinMaxNormalizer():
    def __init__(self,min=0,max=0):
        self.min = min
        self.max = max
    def transform(self,X):
        self.min = np.min(X,axis=2,keepdims=True)
        self.max = np.max(X,axis=2,keepdims=True)

        
        minmax = self.max - self.min
        # minmax[minmax==0] = 1
        minmax[np.abs(minmax) < 1e-10] = 1
        z = (X -self.min) / (minmax)
        return z
    def fit(self,X):
        self.min = np.min(X,axis=2,keepdims=True)
        self.max = np.max(X,axis=2,keepdims=True)
        
        return self

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z 
    
class GlobalMinMaxNormalizer():
    def __init__(self,min=0,max=0):
        self.min = min
        self.max = max
    def transform(self,X):
        self.min = np.min(X,axis=(1,2),keepdims=True)
        self.max = np.max(X,axis=(1,2),keepdims=True)

        
        minmax = self.max - self.min
        # minmax[minmax==0] = 1
        minmax[np.abs(minmax) < 1e-10] = 1
        z = (X -self.min) / (minmax)
        return z
    def fit(self,X):
        self.min = np.min(X,axis=(1,2),keepdims=True)
        self.max = np.max(X,axis=(1,2),keepdims=True)
        
        return self

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

class MeanNormalizer():
    def __init__(self,mean=0,min=0,max=0):
        self.mean = mean
        self.min = min
        self.mean = mean
    def transform(self,X):
        self.mean = np.mean(X,axis=2,keepdims=True)
        self.min = np.min(X,axis=2,keepdims=True)
        self.max = np.max(X,axis=2,keepdims=True)

        minmax = self.max - self.min
        # minmax[minmax==0] = 1
        minmax[np.abs(minmax) < 1e-10] = 1
        z = (X - self.mean) / (minmax)
        return z

    def fit(self,X):
        self.mean = np.mean(X,axis=2,keepdims=True)
        self.min = np.min(X,axis=2,keepdims=True)
        self.max = np.max(X,axis=2,keepdims=True)

        return self

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z 
    
class GlobalMeanNormalizer():
    def __init__(self,mean=0,min=0,max=0):
        self.mean = mean
        self.min = min
        self.mean = mean
    def transform(self,X):
        self.mean = np.mean(X,axis=(1,2),keepdims=True)
        self.min = np.min(X,axis=(1,2),keepdims=True)
        self.max = np.max(X,axis=(1,2),keepdims=True)

        minmax = self.max - self.min
        # minmax[minmax==0] = 1
        minmax[np.abs(minmax) < 1e-10] = 1
        z = (X - self.mean) / (minmax)
        return z

    def fit(self,X):
        self.mean = np.mean(X,axis=(1,2),keepdims=True)
        self.min = np.min(X,axis=(1,2),keepdims=True)
        self.max = np.max(X,axis=(1,2),keepdims=True)

        return self

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

class MedianNormalizer():
    def __init__(self,median=0):
        self.median=median
    def transform(self,X):
        self.median = np.median(X,axis=2,keepdims=True)

        # self.median[self.median == 0] = 1
        self.median[np.abs(self.median) < 1e-10] = 1
        z = X / self.median
        return z

    def fit(self,X):
        self.median = np.median(X,axis=2,keepdims=True)

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z 
    
class GlobalMedianNormalizer():
    def __init__(self,median=0):
        self.median=median
    def transform(self,X):
        self.median = np.median(X,axis=(1,2),keepdims=True)

        # self.median[self.median == 0] = 1
        self.median[np.abs(self.median) < 1e-10] = 1
        z = X / self.median
        return z

    def fit(self,X):
        self.median = np.median(X,axis=(1,2),keepdims=True)

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z
    
class UnitNormalizer():
    def __init__(self):
        pass
    def transform(self,X):
        X_norm = np.linalg.norm(X,axis=2,keepdims=True)
        X_norm[np.abs(X_norm) < 1e-10] = 1
        z = X / X_norm
        return z

    def fit(self,X):
        pass

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z
    
class GlobalUnitNormalizer():
    def __init__(self):
        pass
    def transform(self,X):
        X_norm = np.linalg.norm(X,axis=(1,2),keepdims=True)
        X_norm[np.abs(X_norm) < 1e-10] = 1
        z = X / X_norm
        return z

    def fit(self,X):
        pass

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

class SigmoidNormalizer():
    def __init__(self):
        pass
    def transform(self,X):
        z = sigmoid(X)
        return z

    def fit(self,X):
        pass

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

def tanh(x):
    z = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    return z

class TanhNormalizer():
    def __init__(self):
        pass
    def transform(self,X):
        z = np.tanh(X)
        return z

    def fit(self,X):
        pass

    def fit_transform(self,X):
        self.fit(X)
        z = self.transform(X)
        return z

normalization_methods = {
    'zscore-i':ZNormalizer,
    'zscore-d':GlobalZNormalizer,
    'minmax-i':MinMaxNormalizer,
    'minmax-d':GlobalMinMaxNormalizer,
    'median-i':MedianNormalizer,
    'median-d':GlobalMedianNormalizer,
    'mean-i':MeanNormalizer,
    'mean-d':GlobalMeanNormalizer,
    'unit-i':UnitNormalizer,
    'unit-d':GlobalUnitNormalizer,
    'sigmoid-i':SigmoidNormalizer,
    'tanh-i':TanhNormalizer,
}
    
def create_normalizer(name='zscore',X=None):

    norm = normalization_methods[name]()

    if X is not None:
        X_transform = norm.transform(X)
        return norm, X_transform
    else:
        return norm