"""
Simple comparison: Load ACTUAL ensemble metrics vs Naive
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

print("=" * 80)
print("QUICK CHECK: Naive vs Ensemble (Actual Metrics)")
print("=" * 80)

# Load data
df = pd.read_csv("cleaned_dataset1.csv")
df_cell1 = df[df['CellID'] == 1].copy()
internet = df_cell1['internet'].values

# Split (same as models)
train_size = int(len(internet) * 0.8)
val_size = int(len(internet) * 0.1)
test_start = train_size + val_size

test_data = internet[test_start:]
train_data = internet[:train_size]

print(f"\nTest size: {len(test_data)}")

# NAIVE FORECAST
naive_pred = []
for i in range(len(test_data)):
    if i == 0:
        pred = train_data[-1]
    else:
        pred = test_data[i-1]
    naive_pred.append(pred)

naive_pred = np.array(naive_pred)

naive_mae = mean_absolute_error(test_data, naive_pred)
naive_rmse = math.sqrt(mean_squared_error(test_data, naive_pred))

print(f"\nNAIVE:")
print(f"  MAE:  {naive_mae:.4f}")
print(f"  RMSE: {naive_rmse:.4f}")

# SEASONAL NAIVE
PERIOD = 144
seasonal_pred = []
for i in range(len(test_data)):
    if i == 0:
        history = train_data
    else:
        history = np.concatenate([train_data, test_data[:i]])
    
    if len(history) >= PERIOD:
        pred = history[-PERIOD]
    else:
        pred = history[-1]
    seasonal_pred.append(pred)

seasonal_pred = np.array(seasonal_pred)

seasonal_mae = mean_absolute_error(test_data, seasonal_pred)
seasonal_rmse = math.sqrt(mean_squared_error(test_data, seasonal_pred))

print(f"\nSEASONAL NAIVE:")
print(f"  MAE:  {seasonal_mae:.4f}")
print(f"  RMSE: {seasonal_rmse:.4f}")

print(f"\nBest Baseline: {'Naive' if naive_mae < seasonal_mae else 'Seasonal-Naive'}")
print(f"Best MAE: {min(naive_mae, seasonal_mae):.4f}")

print("\n" + "=" * 80)
print("NOTE: baseline_comparison.py used PLACEHOLDER LSTM values")
print("Real ensemble metrics needed from actual model run")
print("=" * 80)
