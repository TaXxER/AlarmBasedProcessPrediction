from sklearn.base import BaseEstimator
import numpy as np

class LGBMCalibrationWrapper(BaseEstimator):
    def __init__(self, cls):
        self.cls = cls
        self.classes_ = [0,1]
    
    def predict_proba(self, X):
        preds = self.cls.predict(X)
        preds = np.array([1-preds, preds]).T
        return preds
