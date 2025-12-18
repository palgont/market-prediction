# Summary
# -------
# Plot predicted mean + high confidence interval for PPE estimator
# across different orders of the polynomial features
# '''

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("ticks")
# sns.set_context("notebook")

import load_data
from feature_transform import PolynomialFeatureTransform
from bayesian_lr import BayesianLinearRegressionEstimator

# Global
alpha = 0.001


if __name__ == '__main__':
    # N = 20

    x_train_ND, t_train_N, x_val_VD, t_val_V = load_data.load_dataset(
        val_size=10)
    
    N = x_train_ND.shape[0]
    
    # Splice data into balanced time series(no missing values)
    x_train_ND = x_train_ND[6971:]
    t_train_N = t_train_N[6971:]

    figure = 0
    y3 = np.zeros((16, len(t_val_V)))
    order_arr = [1, 5, 9]

    # compute at different orders
    for order in [1, 5, 9]:
        beta_list = [0.2, 1.8, 2.6]
        L = len(beta_list)

        # alpha_list = 0.001 * np.ones(L)
        order_list = order * np.ones(L, dtype=np.int32)
        score_list = []

        for fig_col_id in range(len(order_list)):
            order = order_list[fig_col_id]
            # alpha = alpha_list[fig_col_id]
            beta = beta_list[fig_col_id]

            feature_transformer = PolynomialFeatureTransform(
                    order=order, input_dim = x_train_ND.shape[1])

            # Train Bayesian Linear Regression estimator using first N examples
            ppe_estimator = BayesianLinearRegressionEstimator(feature_transformer, alpha=alpha, beta=beta)
            ppe_estimator.fit(x_train_ND[:N], t_train_N[:N])
            # score on train set
            ppe_tr_score = ppe_estimator.score(x_train_ND[:N], t_train_N[:N])
            # score on test set
            ppe_va_score = ppe_estimator.score(x_val_VD, t_val_V)
            print("order %2d alpha %6.3f beta %6.3f : %8.4f tr score  %8.4f va score" % (
                order, alpha, beta, ppe_tr_score, ppe_va_score))
            score_list.append(ppe_va_score)

    
    # ppe_estimator = BayesianLinearRegressionEstimator(feature_transformer, alpha=alpha, beta=beta)
    # ppe_estimator.fit(x_train_ND, t_train_N)
            y1 = t_val_V
            y2, var_V = ppe_estimator.predict(x_val_VD)
            x = range(len(t_val_V))

            # fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            plt.figure()
            plt.plot(x, y1, label = "Truth")
            plt.plot(x, y2, label = "Predicted")
            plt.legend()
            plt.xlabel("Time Step")
            plt.ylabel("Excess Market Returns")
            plt.title(f"Order: {order}, Beta: {beta:.1f}")

            y3[figure] = y2 - y1
            figure += 1
    
    plt.figure()
    for col_id in range(len(order_arr)):
        for beta in range(len(beta_list)):
            plt.plot(x, y3[len(order_arr)*col_id + beta],
                     label = f"Order:{order_arr[col_id]}, Beta: {beta_list[beta]}")
    
    plt.xlabel("Time Step")
    plt.ylabel("Excess Market Returns Predicted and Truth Difference")
    plt.title("Predicted - Truth: Excess Market Returns")
    plt.legend()
    plt.show()