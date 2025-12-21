import load_data
import preprocess_data
import tree_node
import train_model

nba_data = load_data()

x_train, y_train, x_test, y_test = preprocess_data(nba_data)
data_set_size = x_train.shape[0]

dlt = TreeNode()
train_model(x_train, y_train)

evaluate_model(nba_data)

