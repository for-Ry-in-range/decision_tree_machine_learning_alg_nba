import load_data
import preprocess_data
import tree_node
import train_model
import save_load

# Load and preprocess the data
nba_data = load_data.load_data()
x_train, y_train, x_test, y_test = preprocess_data.preprocess_data(nba_data)
data_set_size = x_train.shape[0]

# Train the model
print("Training the decision tree model...")
dlt = tree_node.TreeNode()
train_model.dfs_train(x_train, y_train, dlt, data_set_size, 0)

# Save the model
save_load.save_model(dlt)
print("Model trained and saved to 'nba_decision_tree.pkl'")

