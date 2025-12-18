# notes
# HMMS must be in time order

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

from feature_transform import PolynomialFeatureTransform

# Allow user to specify location of csv files with DATA_DIR env var
# but by default, use folder structure distributed with source code
src_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('DATA_DIR', 
    os.path.join(os.path.dirname(src_dir), 'kaggle_data'))

# Creating this return data that is in time order for the HMM
def load_dataset_HMM(seed=123, val_size=0, data_dir=DATA_DIR):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    TARGET_COL = 'market_forward_excess_returns'
    ID_COL = 'date_id'

    # Sort by time
    train_df = train_df.sort_values(ID_COL).reset_index(drop=True)
    test_df = test_df.sort_values(ID_COL).reset_index(drop=True)

    DROP_COLS = [ID_COL, TARGET_COL, 'forward_returns', 'risk_free_rate']

    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float) if TARGET_COL in test_df.columns else None

    # (9021, 94) (9021,)
    # y mean/std: 5.321377301777475e-05 0.01055760606125878
    # Any NaNs in X? True
    # Any NaNs in y? False
    # print(X_train.shape, y_train.shape)
    # print("y mean/std:", y_train.mean(), y_train.std())
    # print("Any NaNs in X?", np.isnan(X_train).any())
    # print("Any NaNs in y?", np.isnan(y_train).any())
    if val_size > 0:
        V = int(val_size)
        X_val, y_val = X_train[-V:], y_train[-V:]
        X_train, y_train = X_train[:-V], y_train[:-V]
    else:
        X_val, y_val = None, None

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_dataset_HMM(val_size=0)

# Full training data without NaN values
X_train = X_train[6971:]
y_train = y_train[6971:]

# print(X_train.shape) Output: (2050, 94)
# print(y_train.shape) Output: (2050,)
# print("Any NaNs in X?", np.isnan(X_train).any()) Output: Any NaNs in X? False
# print("Any NaNs in Y?", np.isnan(y_train).any()) Output: Any NaNs in Y? False


# Standardize X (train-only)
# mu = X_train.mean(axis=0)
# sd = X_train.std(axis=0) + 1e-8

# X_train = (X_train - mu) / sd
# X_val   = (X_val   - mu) / sd
# X_test  = (X_test  - mu) / sd

from hmm_gaussian import GaussianHMM

# Fit HMM on y only (ignore X for now)
model = GaussianHMM(K=3, n_iter=50, seed=0)
model.fit(y_train, verbose=True)

print("\nLearned pi:", model.pi)
print("\nLearned A:\n", model.A)
print("\nLearned mu:", model.mu)
print("\nLearned sigma:", np.sqrt(model.var))

states = model.viterbi(y_train)
print("\nState counts:", np.bincount(states))

import matplotlib.pyplot as plt

states = model.viterbi(y_train)

plt.figure()
plt.plot(y_train, linewidth=1)
plt.title("Forward Excess Returns (y)")
plt.show()

plt.figure()
plt.plot(states, linewidth=1)
plt.title("Viterbi states over time")
plt.yticks([0,1,2])
plt.show()






