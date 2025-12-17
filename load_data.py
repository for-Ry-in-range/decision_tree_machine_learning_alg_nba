import numpy as np
import pandas as pd

data = pd.read_csv('nba_data.csv')

y = data['target_5yrs']  # Target column

x = data.drop(columns=['name', 'target_5yrs']).values  # Remove target and name columns

