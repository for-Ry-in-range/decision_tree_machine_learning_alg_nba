def evaluate(x, y, root):
    
    summed_scores = 0

    def dfs(cur, row, parent_node):

        # Check if node exists
        if not cur:
            return parent_node.prediction if parent_node else 0
        
        # If this is a leaf node
        if cur.split_1 is None or cur.split_2 is None or cur.split_3 is None:
            return cur.prediction

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

        # Check if node exists
        if not cur:
            if parent_node:
                return parent_node.prediction
            else:
                return 0
        
        # If this is a leaf node
        if cur.split_1 is None or cur.split_2 is None or cur.split_3 is None:
            return cur.prediction

        if row[cur.feature] < cur.split_1:
            return dfs(cur.child_a, row, cur)
        if row[cur.feature] < cur.split_2:
            return dfs(cur.child_b, row, cur)
        if row[cur.feature] < cur.split_3:
            return dfs(cur.child_c, row, cur)
        else:
            return dfs(cur.child_d, row, cur)
    
    return dfs(root, row, None)