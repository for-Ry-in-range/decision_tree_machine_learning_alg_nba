import pandas as pd
import save_load
import evaluate_model
import sys

# Check if filename was provided
if len(sys.argv) < 2:
    print("Usage: python3 predict.py <csv_filename>")
    sys.exit(1)

# Get the filename from command line argument
new_player_file = sys.argv[1]

# Load the trained model
print("Loading trained model...")
model = save_load.load_model()

# Load new player data
print(f"Loading new player data from '{new_player_file}'...")
try:
    new_player_data = pd.read_csv(new_player_file)
except FileNotFoundError:
    print(f"File '{new_player_file}' not found")
    sys.exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

new_player_features = new_player_data.drop(columns=['name', 'target_5yrs'], errors='ignore')

# Make predictions
print("Making predictions...")
for i in range(len(new_player_features)):
    row = new_player_features.iloc[i]
    player_name = new_player_data.iloc[i]['name']
    
    prediction = evaluate_model.predict_single(model, row)
    if prediction == 1:
        result = "5+ years" 
    else:
        result = "Less than 5 years"
    
    print(f"{player_name}: {result}")

