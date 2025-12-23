import pandas as pd
import numpy as np
from tree_node import TreeNode


def decide_split(col, col_name, y):
    col.sort_values(by=col_name)
    data_size = y.shape[0]-1
    min_entropy = 1.1
    best_midpoint = None

    # Go through each midpoint betwen values
    for i in range(col.shape[0]-1):
        before = col.iloc[i][col_name]
        after = col.iloc[i+1][col_name]
        mid = (before + after) / 2

        entropy = None
        greater_than = 0
        less_than_yes = 0
        greater_than_yes = 0

        # Get entropy for this midpoint
        for j in range(col.shape[0]):
            if col.iloc[j][col_name] >= mid:
                greater_than += 1
                if y.iloc[j]['target_5yrs'] == 1.0:
                    greater_than_yes += 1
            else:
                if y.iloc[j]['target_5yrs'] == 1.0:
                    less_than_yes += 1
            
            entropy = None
            if greater_than == 0 or greater_than == data_size:
                entropy = 0
            else:
                entropy = (greater_than/data_size * (-1*(greater_than_yes/greater_than * np.log2(greater_than_yes/greater_than)) + ((1 - greater_than_yes/greater_than) * np.log2(1 - greater_than_yes/greater_than))) + (1-greater_than/data_size) * (-1*(less_than_yes/(data_size-greater_than) * np.log2(less_than_yes/(data_size-greater_than))) + ((1 - less_than_yes/(data_size-greater_than)) * np.log2(1 - less_than_yes/(data_size-greater_than)))))
        if entropy < min_entropy:
            min_entropy = entropy
            best_midpoint = mid
        return best_midpoint
        
def separate_rows_around_midpoint(midpoint, col, col_name, y):
    # Select rows based on the split
    greater_than_rows = col[col_name] >= midpoint
    less_than_rows = col[col_name] < midpoint
    return greater_than_rows, less_than_rows


def calculate_entropy(y):
    size = y.shape[0]
    made_5_years = 0
    for val in y['target_5yrs']:
        if val == 1.0:
            made_5_years += 1
    if size == 0:
        return 0
    return -1 * (made_5_years/size * (np.log2(made_5_years/size)) + (1-made_5_years/size) * (np.log2(1-made_5_years/size)))


def dfs_train(x, y, cur_node, data_set_size):
    if x.shape[1] == 0 or data_set_size == 0:
        return

    min_feature_entropy = 1.1
    feature = None
    min_feat_a_rows = None
    min_feat_b_rows = None
    min_feat_c_rows = None
    min_feat_d_rows = None
    min_feat_a_b_rows = None
    min_feat_c_d_rows = None

    # Check information gain of each column
    for col in x.columns:
        # Split into two parts
        split_2 = decide_split(x[[col]], col, y)
        c_d_rows, a_b_rows = separate_rows_around_midpoint(split_2, x[[col]], col, y)
        
        # Split by rows
        greater_than_x = x[c_d_rows]
        less_than_x = x[a_b_rows]
        greater_than_y = y[c_d_rows]
        less_than_y = y[a_b_rows]

        # Split into four parts in total
        split_1 = decide_split(less_than_x[[col]], col, less_than_y)
        split_3 = decide_split(greater_than_x[[col]], col, greater_than_y)
        
        # Get all separated rows - first letter is section, second letter is input/output
        b_rows, a_rows = separate_rows_around_midpoint(split_1, less_than_x[[col]], col, less_than_y)
        b_x = less_than_x[b_rows]
        a_x = less_than_x[a_rows]
        b_y = less_than_y[b_rows]
        a_y = less_than_y[a_rows]

        d_rows, c_rows = separate_rows_around_midpoint(split_3, greater_than_x[[col]], col, greater_than_y)
        d_x = greater_than_x[d_rows]
        c_x = greater_than_x[c_rows]
        d_y = greater_than_y[d_rows]
        c_y = greater_than_y[c_rows]        

        # Calculate information gain
        a_entropy_with_weight = a_x.shape[0]/data_set_size * calculate_entropy(a_y)
        b_entropy_with_weight = b_x.shape[0]/data_set_size * calculate_entropy(b_y)
        c_entropy_with_weight = c_x.shape[0]/data_set_size * calculate_entropy(c_y)
        d_entropy_with_weight = d_x.shape[0]/data_set_size * calculate_entropy(d_y)
        cur_feature_entropy = a_entropy_with_weight + b_entropy_with_weight + c_entropy_with_weight + d_entropy_with_weight

        # Set new entropy leader
        if cur_feature_entropy < min_feature_entropy:
            feature = col
            min_feature_entropy = cur_feature_entropy
            min_feat_a_rows = a_rows
            min_feat_b_rows = b_rows
            min_feat_c_rows = c_rows
            min_feat_d_rows = d_rows
            min_feat_a_b_rows = a_b_rows
            min_feat_c_d_rows = c_d_rows

    # Set current node's values
    cur_node.feature = feature
    cur_node.child_a = TreeNode()
    cur_node.child_b = TreeNode()
    cur_node.child_c = TreeNode()
    cur_node.child_d = TreeNode()

    # Remove this feature
    a_b_x = x[min_feat_a_b_rows]
    c_d_x = x[min_feat_c_d_rows]
    a_b_y = y[min_feat_a_b_rows]
    c_d_y = y[min_feat_c_d_rows]
    inputs = [a_b_x[min_feat_a_rows], a_b_x[min_feat_b_rows], c_d_x[min_feat_c_rows], c_d_x[min_feat_d_rows]]
    for input in inputs:
        input.drop(columns=[feature])

    # Recursively create the next branches
    dfs_train(inputs[0], a_b_y[min_feat_a_rows], cur_node.child_a, inputs[0].shape[0])
    dfs_train(inputs[1], a_b_y[min_feat_b_rows], cur_node.child_b, inputs[1].shape[0])
    dfs_train(inputs[2], c_d_y[min_feat_c_rows], cur_node.child_c, inputs[2].shape[0])
    dfs_train(inputs[3], c_d_y[min_feat_d_rows], cur_node.child_d, inputs[3].shape[0])
