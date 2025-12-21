import pandas as pd
import numpy as np
import main
from tree_node import TreeNode


def decide_split(col, col_name, y):
    col.sort_values(col_name)
    data_size = y.shape[0]-1
    min_entropy = 1.1
    best_midpoint = None

    # Go through each midpoint betwen values
    for i in range(col[col_name].shape[0]-1):
        before = col.iloc[i][col_name]
        after = col.iloc[i+1][col_name]
        mid = (before + after) / 2

        entropy = None
        greater_than = 0
        less_than_yes = 0
        greater_than_yes = 0

        # Get entropy for this midpoint
        for j in range(col[col_name].shape[0]):
            if col.iloc[j][col_name] >= mid:
                greater_than += 1
                if y.iloc[j] == 1.0:
                    greater_than_yes += 1
            else:
                if y.iloc[j] == 1.0:
                    less_than_yes += 1
            
            entropy = (greater_than/data_size * (-1*(greater_than_yes/greater_than * np.log2(greater_than_yes/greater_than)) + ((1 - greater_than_yes/greater_than) * np.log2(1 - greater_than_yes/greater_than))) + (1-greater_than/data_size) * (-1*(less_than_yes/(data_size-greater_than) * np.log2(less_than_yes/(data_size-greater_than))) + ((1 - less_than_yes/(data_size-greater_than)) * np.log2(1 - less_than_yes/(data_size-greater_than)))))
        if entropy < min_entropy:
            min_entropy = entropy
            best_midpoint = mid
        return best_midpoint
        
def separate_around_midpoint(midpoint, col, col_name, y):
    # Select rows based on the split
    greater_than_rows = col[col_name] >= midpoint
    less_than_rows = col[col_name] < midpoint
    
    # Split X
    greater_than_x = col[greater_than_rows]
    less_than_x = col[less_than_rows]
    
    # Split Y
    greater_than_y = y[greater_than_rows]
    less_than_y = y[less_than_rows]
    
    return less_than_x, greater_than_x, less_than_y, greater_than_y


def calculate_entropy(y):
    size = y.shape[0]
    made_5_years = 0
    for val in y:
        if val == 1.0:
            made_5_years += 1
    return -1 * (made_5_years/size * (np.log2(made_5_years/size)) + (1-made_5_years/size) * (np.log2(1-made_5_years/size)))


def dfs(x, y, cur_node):
    min_feature_entropy = 1.1
    feature = None

    # Check information gain of each column
    for col in x.columns:
        # Split into two parts
        split_2 = decide_split(x[col], col, y)
        less_than_x, greater_than_x, less_than_y, greater_than_y = separate_around_midpoint(split_2, x[col], col, y)
        
        # Split into four parts in total
        split_1 = decide_split(less_than_x[col], col, less_than_y)
        split_3 = decide_split(greater_than_x[col], col, greater_than_y)
        
        # Get all separated rows - first letter is section, second letter is input/output
        a_x, b_x, a_y, b_y = separate_around_midpoint(split_1, less_than_x[col], col, less_than_y)
        c_x, d_x, c_y, d_y = separate_around_midpoint(split_3, greater_than_x[col], col, greater_than_y)

        # Calculate information gain
        a_entropy_with_weight = a_x.shape[0]/main.data_set_size * calculate_entropy(a_y)
        b_entropy_with_weight = b_x.shape[0]/main.data_set_size * calculate_entropy(b_y)
        c_entropy_with_weight = c_x.shape[0]/main.data_set_size * calculate_entropy(c_y)
        d_entropy_with_weight = d_x.shape[0]/main.data_set_size * calculate_entropy(d_y)
        cur_feature_entropy = a_entropy_with_weight + b_entropy_with_weight + c_entropy_with_weight + d_entropy_with_weight

        # Set new entropy leader
        if cur_feature_entropy < min_feature_entropy:
            feature = col
            min_feature_entropy = cur_feature_entropy

    cur_node.feature = feature
    cur_node.child_a = TreeNode()
    cur_node.child_b = TreeNode()
    cur_node.child_c = TreeNode()
    cur_node.child_d = TreeNode()

    dfs(a_x, a_y, cur_node.child_a)
    dfs(b_x, b_y, cur_node.child_b)
    dfs(c_x, c_y, cur_node.child_c)
    dfs(d_x, d_y, cur_node.child_d)


def train_model(x, y):
    x_copy = x.copy()  # Copy so that the original data set is not modified
    dfs(x_copy, y, main.dlt)
    