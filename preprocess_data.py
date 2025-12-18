import numpy as np

def preprocess_data(nba_data):
    y = nba_data['target_5yrs']  # Target column
    x = nba_data.drop(columns=['name', 'target_5yrs']).values  # Remove target and name columns
    
    # Randomly shuffle row order
    rows_indices = np.arrange(x.shape[0])
    np.random.shuffle(rows_indices)

    # Switch rows into shuffled order
    x = x[rows_indices]
    y = y[rows_indices]

    # Split data into training and testing sets
    split_index = int(0.8 * x.shape[0])
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]

    return x_train, y_train, x_test, y_test