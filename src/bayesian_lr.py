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

class BayesianLinearRegressionEstimator():

    def __init__(self, feature_transformer, alpha=1.0, beta=1.0):
        self.feature_transformer = feature_transformer
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.M = self.feature_transformer.get_feature_size() # num features        

    # fit
    # input: x_ND, t_N
    # output: GMM with fitted mean, covariance, and precision
    # summary: Define distributions modeling each weight(coefficient) of all
    #          features in transformed input, phi_NM.
    def fit(self, x_ND, t_N):
        phi_NM = self.feature_transformer.transform(x_ND)

        self.precision_MM = self.alpha * np.eye(self.M) + self.beta * (phi_NM.T @ phi_NM)
        self.covariance_MM = np.linalg.inv(self.precision_MM)
        # 
        self.mean_M = self.beta * self.covariance_MM @ (phi_NM.T @ t_N)

        return self


    # predict
    # input: x_VD
    # output: predictions of output values given some new input feature vectors
    #         p(t* | x*, D)
    # summary: Predict scalar mean and var given test input
    def predict(self, x_VD):
        phi_NM = self.feature_transformer.transform(x_VD)

        mean_V = phi_NM @ self.mean_M
        var_V = (1.0 / self.beta) + np.sum(phi_NM @ self.covariance_MM * phi_NM, axis = 1)
        
        return mean_V, var_V


    # score
    # CP2
    # input: input raw feature vector x_ND, correct output values for train set, t_n
    # output: average log probability across all weights
    # summary: Compute the average posterior predictive log likelihood of the
    #          a given the real input, output pair
    def score(self, x_ND, t_N):
            ''' Compute the average log likelihoods of provided dataset
        
            Assumes w is set to MAP value (internal attribute).
            Assumes Normal iid likelihood with precision \beta.

            Args
            ----
            x_ND : 2D array, shape (N, D)
                Each row is a 'raw' feature vector of size D
                D is same as self.feature_transformer.input_dim
            t_N : 1D array, shape (N,)
                Outputs for regression

            Returns
            -------
            avg_log_proba : float
            '''
            phi_NM = self.feature_transformer.transform(x_ND)
            mean_N = phi_NM @ self.mean_M

            var_N = (1.0 / self.beta) + np.sum(
                 phi_NM @ self.covariance_MM * phi_NM, axis = 1
            )
            sd_N = np.sqrt(var_N)

            total_log_probs = scipy.stats.norm.logpdf(
                t_N, mean_N, sd_N)
            
            return np.mean(total_log_probs)
