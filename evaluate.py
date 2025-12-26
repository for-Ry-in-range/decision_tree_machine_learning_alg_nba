import load_data
import preprocess_data
import save_load
import evaluate_model

# Load and preprocess the data
nba_data = load_data.load_data()
x_train, y_train, x_test, y_test = preprocess_data.preprocess_data(nba_data)

# Get the trained model
print("Loading trained model...")
model = save_load.load_model()

# Evaluate the model
print("Evaluating model on test data...")
score = evaluate_model.evaluate(x_test, y_test, model)
print(f"Model Accuracy: {score}")

