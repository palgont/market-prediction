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

def load_dataset(seed=123, val_size=0, data_dir=DATA_DIR):
    # Load and unpack training and test data
    train_csv_fpath = os.path.join(data_dir, 'train.csv')
    test_csv_fpath = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(train_csv_fpath):
        raise FileNotFoundError("Please set DATA_DIR. Cannot find CSV files at path: ",
            train_csv_fpath)

    # train_df = pd.read_csv(train_csv_fpath)
    # test_df = pd.read_csv(test_csv_fpath)
    # x_train_ND = train_df['x'].values[:,np.newaxis]
    # t_train_N = train_df['y'].values

    # random_state = np.random.RandomState(int(seed))
    # shuffle_ids = random_state.permutation(t_train_N.size)
    # x_train_ND = x_train_ND[shuffle_ids]
    # t_train_N = t_train_N[shuffle_ids]

    # x_test_ND = test_df['x'].values[:,np.newaxis]
    # t_test_N = test_df['y'].values

    train_df = pd.read_csv(train_csv_fpath)
    test_df  = pd.read_csv(test_csv_fpath)

    # Choose target
    TARGET_COL = 'market_forward_excess_returns'   # <- most likely what you want
    ID_COL = 'date_id'

    # Features = everything except id + target(s)
    DROP_COLS = [ID_COL, TARGET_COL, 'forward_returns', 'risk_free_rate']
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

    # Build arrays
    x_train_ND = train_df[feature_cols].to_numpy()
    t_train_N  = train_df[TARGET_COL].to_numpy()

    # Test labels might or might not exist depending on competition setup.
    # If test.csv has no target column, set t_test_N = None
    x_test_ND = test_df[feature_cols].to_numpy()
    t_test_N  = test_df[TARGET_COL].to_numpy() if TARGET_COL in test_df.columns else None

    if val_size == 0:
        return x_train_ND, t_train_N, x_test_ND, t_test_N
    else:
        assert val_size > 0
        V = int(val_size)
        x_val_VD, t_val_V = x_train_ND[-V:], t_train_N[-V:]
        x_train_ND, t_train_N = x_train_ND[:-V], t_train_N[:-V]
        return x_train_ND, t_train_N, x_val_VD, t_val_V