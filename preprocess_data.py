import numpy as np
import pandas as pd

def preprocess_data(nba_data):
    y = nba_data[['target_5yrs']]  # Target column (DataFrame - using double brackets)
    x = nba_data.drop(columns=['name', 'target_5yrs'])  # Remove target and name columns (DataFrame)
    
    # Randomly shuffle row order
    rows_indices = np.arange(x.shape[0])
    np.random.shuffle(rows_indices)

    # Switch rows into shuffled order; using iloc to keep vars as dataframes
    x = x.iloc[rows_indices].reset_index(drop=True)
    y = y.iloc[rows_indices].reset_index(drop=True)

    # Split data into training and testing sets
    split_index = int(0.8 * x.shape[0])
    x_train = x.iloc[:split_index].reset_index(drop=True)
    y_train = y.iloc[:split_index].reset_index(drop=True)
    x_test = x.iloc[split_index:].reset_index(drop=True)
    y_test = y.iloc[split_index:].reset_index(drop=True)

    return x_train, y_train, x_test, y_test