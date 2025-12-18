import load_data
import preprocess_data

nba_data = load_data()

x_train, y_train, x_test, y_test = preprocess_data(nba_data)

train_model(nba_data)

evaluate_model(nba_data)

