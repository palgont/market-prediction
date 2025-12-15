""""
Summary
-------
Defines the posterior bayesian linear regression for mvnormpdf

BLREstimator supports these API functions common to any sklearn-like regression model:
* fit
* predict
* score

Resources 
---------
CP1 Assignment Code
"""
import numpy as np
import scipy.stats

# Outline for ppe class
"""
class LinearRegressionEstimator():
    def __init__(self, feature_transformer, alpha=1.0, beta=1.0):
    
    def fit(self, x_ND, t_N):
    
    def predict(self, x_ND):
    
    def predict_variance(self, x_ND):
    
    def score(self, x_ND, t_N):
    
    def fit_and_calc_log_evidence(self, x_ND, t_N):
"""

def __init__(self, feature_transformer, alpha=1.0, beta=1.0):
    self.feature_transformer = feature_transformer
    self.alpha = float(alpha)
    self.beta = float(beta)
    self.M = self.feature_transformer.get_feature_size() # num features        

# fit
# input: x_ND, 
# output: GMM with fitted mean, covariance, and precision
def fit(self, x_ND, t_N):
    phi_NM = self.feature_transformer.transform(x_ND)

    self.precision_MM = self.alpha * np.eye(self.M) + self.beta * (phi_NM.T @ phi_NM)
    self.covariance_MM = np.linalg.inv(self.precision_MM)
    # 
    self.mean_M = self.beta * self.covariance_MM @ (phi_NM.T @ t_N)

    return self


# predict
# input: x_ND
# output: predictions for output values of input feature vectors
def predict(self, x_ND):
    phi_NM = self.feature_transformer.transform(x_ND)
    N, M = phi_NM.shape
    
    t_est_N = phi_NM @ self.mean_M

    return t_est_N



# score
