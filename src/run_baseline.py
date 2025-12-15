# Summary
# -------
# Plot predicted mean + high confidence interval for PPE estimator
# across different orders of the polynomial features
# '''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("notebook")

import load_data
# from feature_transform import PolynomialFeatureTransform
# from bayesian_lr import LinearRegressionPosteriorPredictiveEstimator


if __name__ == '__main__':
    N = 20   
    x_train_ND, t_train_N, x_val_VD, t_val_V = load_data.load_dataset(
        val_size=100)
    
    print(x_train_ND)

    # # Polynomial orders to try
    # for order in [1, 4, 7]:

    #     # Set precisions to try for likelihood
    #     beta_list = [0.2, 0.6, 1.8]
    #     L = len(beta_list)

    #     # Set precisions of prior (alpha)
    #     alpha_list = 0.001 * np.ones(L)
    #     order_list = order * np.ones(L, dtype=np.int32)
    #     score_list = []

    #         # Train PPE estimator using only first N examples
    #         ppe_estimator = LinearRegressionPosteriorPredictiveEstimator(
    #             feature_transformer, alpha=alpha, beta=beta)
    #         ppe_estimator.fit(x_train_ND[:N], t_train_N[:N])
    #         ppe_tr_score = ppe_estimator.score(x_train_ND[:N], t_train_N[:N])
    #         ppe_va_score = ppe_estimator.score(x_val_VD, t_val_V)
    #         print("order %2d alpha %6.3f beta %6.3f : %8.4f tr score  %8.4f va score" % (
    #             order, alpha, beta, ppe_tr_score, ppe_va_score))
    #         score_list.append(ppe_va_score)

    #         # Obtain predicted mean and stddev for PPE estimator
    #         # at each x value in provided dense grid of size G
    #         ppe_mean_G = ppe_estimator.predict(xgrid_G1)
    #         ppe_var_G = ppe_estimator.predict_variance(xgrid_G1)
    #         ppe_stddev_G = np.sqrt(ppe_var_G)

    #         cur_ppe_ax = ppe_axgrid[0, fig_col_id]
    #         regr_viz_utils.plot_predicted_mean_with_filled_stddev_interval(
    #             cur_ppe_ax, # plot on PPE figure's current axes
    #             xgrid_G1, ppe_mean_G, ppe_stddev_G,
    #             num_stddev=2,
    #             color='b',
    #             legend_label='PPE +/- 2 stddev')

    #     regr_viz_utils.finalize_x_vs_t_plot(
    #         ppe_axgrid, x_train_ND[:N], t_train_N[:N], x_val_VD, t_val_V,
    #         order_list, alpha_list, beta_list, score_list)
    #     plt.savefig("fig1d_order%02d_viz_predictions.jpg" % order,
    #         bbox_inches='tight', pad_inches=0)
    #     plt.show()