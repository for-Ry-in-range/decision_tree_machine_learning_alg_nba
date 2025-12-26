def save_model(root):
    with open('nba_decision_tree.pkl', 'wb') as f:
        pickle.dump(root, f)

def load_model():
    with open('nba_decision_tree.pkl', 'rb') as f:
        return pickle.load(f)