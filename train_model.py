import pandas as pd
import numpy as np
from tree_node import TreeNode


def decide_split(col, col_name, y):
    col = col.sort_values(by=col_name).reset_index(drop=True)
    y = y.reset_index(drop=True)
    data_size = col.shape[0]
    min_entropy = float('inf')
    best_midpoint = None

    # Go through each midpoint betwen values
    for i in range(data_size - 1):
        before = col.iloc[i][col_name]
        after = col.iloc[i + 1][col_name]
        mid = (before + after) / 2

        # Get indices
        greater_idx = col[col[col_name] >= mid].index
        less_idx = col[col[col_name] < mid].index

        # Counts
        greater_than = len(greater_idx)
        less_than = len(less_idx)
        greater_than_yes = y.loc[greater_idx, 'target_5yrs'].sum()
        less_than_yes = y.loc[less_idx, 'target_5yrs'].sum()

        # Check division by zero
        if greater_than == 0 or less_than == 0:
            entropy = 0
        else:
            p_gt = greater_than_yes / greater_than if greater_than > 0 else 0
            p_lt = less_than_yes / less_than if less_than > 0 else 0

            def entropy_part(p):
                if p == 0 or p == 1:
                    return 0
                return -1 * p * np.log2(p) - (1 - p) * np.log2(1 - p)

            entropy = (greater_than / data_size) * entropy_part(p_gt) + (less_than / data_size) * entropy_part(p_lt)

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
    if size == 0 or made_5_years == 0 or 1-made_5_years/size == 0:
        return 0
    return -1 * (made_5_years/size * (np.log2(made_5_years/size)) + (1-made_5_years/size) * (np.log2(1-made_5_years/size)))


def dfs_train(x, y, cur_node, data_set_size, depth):
    depth += 1
    if x.shape[1] == 0 or data_set_size == 0:
        return

    min_feature_entropy = 1.1
    feature = None
    min_feat_a_x = None
    min_feat_b_x = None
    min_feat_c_x = None
    min_feat_d_x = None
    min_feat_a_y = None
    min_feat_b_y = None
    min_feat_c_y = None
    min_feat_d_y = None
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
            min_feat_a_x = a_x
            min_feat_b_x = b_x
            min_feat_c_x = c_x
            min_feat_d_x = d_x
            min_feat_a_y = a_y
            min_feat_b_y = b_y
            min_feat_c_y = c_y
            min_feat_d_y = d_y

    # Set current node's values
    cur_node.feature = feature
    cur_node.child_a = TreeNode()
    cur_node.child_b = TreeNode()
    cur_node.child_c = TreeNode()
    cur_node.child_d = TreeNode()

    if depth == 4:
        return

    # Remove this feature from each subset
    min_feat_a_x = min_feat_a_x.drop(columns=[feature])
    min_feat_b_x = min_feat_b_x.drop(columns=[feature])
    min_feat_c_x = min_feat_c_x.drop(columns=[feature])
    min_feat_d_x = min_feat_d_x.drop(columns=[feature])

    # Recursively create the next branches
    dfs_train(min_feat_a_x, min_feat_a_y, cur_node.child_a, min_feat_a_x.shape[0], depth)
    dfs_train(min_feat_b_x, min_feat_b_y, cur_node.child_b, min_feat_b_x.shape[0], depth)
    dfs_train(min_feat_c_x, min_feat_c_y, cur_node.child_c, min_feat_c_x.shape[0], depth)
    dfs_train(min_feat_d_x, min_feat_d_y, cur_node.child_d, min_feat_d_x.shape[0], depth)
