def evaluate(x, y, root):
    
    summed_scores = 0

    def dfs(cur, row, parent_node):

        # If this is the last node in the DLT
        if not cur.split_1 or not cur.split_2 or not cur.split_3:
            return cur.prediction
        if not cur:
            return parent_node.prediction

        if row[cur.feature] < cur.split_1:
            return dfs(cur.child_a, row, cur)
        if row[cur.feature] < cur.split_2:
            return dfs(cur.child_b, row, cur)
        if row[cur.feature] < cur.split_3:
            return dfs(cur.child_c, row, cur)
        else:
            return dfs(cur.child_d, row, cur)

    # Iterate through each row of x and y
    for i in range(len(x)):
        prediction = dfs(root, x.iloc[i], None)
        if prediction == y.iloc[i]['target_5yrs']:
            summed_scores += 1

    return summed_scores / len(y)


def predict_single(root, row):
    """
    Make a prediction for one row of rookie data
    """
    def dfs(cur, row, parent_node):
        
        # If this is the last node in the DLT
        if not cur.split_1 or not cur.split_2 or not cur.split_3:
            return cur.prediction
        if not cur:
            return parent_node.prediction

        if row[cur.feature] < cur.split_1:
            return dfs(cur.child_a, row, cur)
        if row[cur.feature] < cur.split_2:
            return dfs(cur.child_b, row, cur)
        if row[cur.feature] < cur.split_3:
            return dfs(cur.child_c, row, cur)
        else:
            return dfs(cur.child_d, row, cur)
    
    return dfs(root, row, None)