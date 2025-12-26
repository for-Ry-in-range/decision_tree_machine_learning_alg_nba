import load_data
import preprocess_data
import tree_node
import train_model
import pickle

nba_data = load_data.load_data()

x_train, y_train, x_test, y_test = preprocess_data.preprocess_data(nba_data)
data_set_size = x_train.shape[0]

dlt = tree_node.TreeNode()
train_model.dfs_train(x_train, y_train, dlt, data_set_size, 0)

pickle.save_model(save_model(dlt))

model = pickle.load_model()

# Evaluate model

# Make prediction on new data