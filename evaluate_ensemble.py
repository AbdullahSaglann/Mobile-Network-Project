"""
Load saved ensemble models and evaluate on test set
Get ACTUAL metrics for comparison
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

print("=" * 80)
print("📊 ENSEMBLE MODEL EVALUATION - ACTUAL METRICS")
print("=" * 80)

# Load data
df = pd.read_csv("cleaned_dataset1.csv")
df_cell1 = df[df['CellID'] == 1].copy()
df_cell1['datetime'] = pd.to_datetime(df_cell1['datetime'])
df_cell1 = df_cell1.sort_values('datetime').reset_index(drop=True)

# Feature engineering
print("\n🎯 Feature engineering...")
df_cell1['hour'] = df_cell1['datetime'].dt.hour
df_cell1['is_peak_morning'] = df_cell1['hour'].isin([8, 9, 10]).astype(int)
df_cell1['is_peak_evening'] = df_cell1['hour'].isin([17, 18, 19, 20]).astype(int)
df_cell1['is_peak'] = (df_cell1['is_peak_morning'] | df_cell1['is_peak_evening']).astype(int)

df_cell1['internet_lag_1h'] = df_cell1['internet'].shift(6)
df_cell1['internet_lag_2h'] = df_cell1['internet'].shift(12)
df_cell1['internet_lag_6h'] = df_cell1['internet'].shift(36)
df_cell1['internet_change_1h'] = df_cell1['internet'] - df_cell1['internet_lag_1h']
df_cell1['is_increasing'] = (df_cell1['internet_change_1h'] > 0).astype(int)

ROLL_WINDOW = 6
df_cell1["internet_roll_mean"] = df_cell1["internet"].rolling(ROLL_WINDOW).mean()
df_cell1["internet_roll_std"] = df_cell1["internet"].rolling(ROLL_WINDOW).std()
df_cell1["internet_roll_max"] = df_cell1["internet"].rolling(ROLL_WINDOW).max()
df_cell1["internet_roll_min"] = df_cell1["internet"].rolling(ROLL_WINDOW).min()
df_cell1["internet_roll_75p"] = df_cell1["internet"].rolling(ROLL_WINDOW).quantile(0.75)
df_cell1["internet_roll_25p"] = df_cell1["internet"].rolling(ROLL_WINDOW).quantile(0.25)
df_cell1["is_weekend"] = (df_cell1["datetime"].dt.weekday >= 5).astype(int)

df_cell1.fillna(method="bfill", inplace=True)
df_cell1.fillna(0, inplace=True)

FEATURE_COLS = [
    "smsin", "smsout", "callin", "callout", "internet",
    "day_sin", "day_cos", "hour_sin", "hour_cos",
    "is_weekend", "is_peak", "is_peak_morning", "is_peak_evening",
    "internet_lag_1h", "internet_lag_2h", "internet_lag_6h",
    "internet_change_1h", "is_increasing",
    "internet_roll_mean", "internet_roll_std",
    "internet_roll_max", "internet_roll_min",
    "internet_roll_75p", "internet_roll_25p",
]

data_feat = df_cell1[FEATURE_COLS].values
data_target = df_cell1["internet"].values

# Split (same as training)
train_size = int(len(df_cell1) * 0.8)
val_size = int(len(df_cell1) * 0.1)

feat_train = data_feat[:train_size]
feat_val = data_feat[train_size:train_size + val_size]
feat_test = data_feat[train_size + val_size:]

target_train = data_target[:train_size].reshape(-1, 1)
target_val = data_target[train_size:train_size + val_size].reshape(-1, 1)
target_test = data_target[train_size + val_size:].reshape(-1, 1)

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_raw = scaler_X.fit_transform(feat_train)
X_test_raw = scaler_X.transform(feat_test)

y_train_raw = scaler_y.fit_transform(target_train)
y_test_raw = scaler_y.transform(target_test)

# Create sequences
W_PAST = 48
H_FUTURE = 6

def create_sequences(data_feat, data_target, w_past, h_future):
    X, y = [], []
    for i in range(len(data_feat) - w_past - h_future + 1):
        X.append(data_feat[i : i + w_past])
        y.append(data_target[i + w_past : i + w_past + h_future])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(X_test_raw, y_test_raw, W_PAST, H_FUTURE)

print(f"\nTest sequences shape: X={X_test.shape}, y={y_test.shape}")

# Load models
print("\n📦 Loading ensemble models...")
try:
    classifier = keras.models.load_model("ensemble_peak_classifier.h5")
    normal_lstm = keras.models.load_model("ensemble_normal_lstm.h5")
    peak_lstm = keras.models.load_model("ensemble_peak_lstm.h5")
    print("✓ All 3 models loaded!")
    
    # Predict
    print("\n🔮 Generating predictions...")
    peak_prob = classifier.predict(X_test, verbose=0).flatten()
    normal_pred = normal_lstm.predict(X_test, verbose=0)
    peak_pred = peak_lstm.predict(X_test, verbose=0)
    
    # Ensemble
    ensemble_pred = peak_prob.reshape(-1, 1, 1) * peak_pred + (1 - peak_prob.reshape(-1, 1, 1)) * normal_pred
    
    # Inverse transform
    y_test_flat = y_test.reshape(-1, 1)
    ensemble_flat = ensemble_pred.reshape(-1, 1)
    
    y_test_final = scaler_y.inverse_transform(y_test_flat)
    ensemble_final = scaler_y.inverse_transform(ensemble_flat)
    
    # Metrics
    mse = mean_squared_error(y_test_final, ensemble_final)
    mae = mean_absolute_error(y_test_final, ensemble_final)
    rmse = math.sqrt(mse)
    
    mask = y_test_final.flatten() != 0
    mape = np.mean(np.abs((y_test_final.flatten()[mask] - ensemble_final.flatten()[mask]) / y_test_final.flatten()[mask])) * 100
    
    print("\n" + "=" * 80)
    print("✅ ENSEMBLE ACTUAL RESULTS")
    print("=" * 80)
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("=" * 80)
    
    # Save to file for baseline comparison
    with open("ensemble_metrics.txt", "w") as f:
        f.write(f"MAE={mae:.4f}\n")
        f.write(f"RMSE={rmse:.4f}\n")
        f.write(f"MAPE={mape:.2f}\n")
    
    print("\n✓ Metrics saved to ensemble_metrics.txt")
    
    # Quick baseline comparison
    print("\n" + "=" * 80)
    print("📊 QUICK BASELINE COMPARISON")
    print("=" * 80)
    
    # Naive baseline on same test data
    test_data_raw = target_test.flatten()
    train_data_raw = target_train.flatten()
    
    naive_pred = []
    for i in range(len(test_data_raw)):
        if i == 0:
            pred = train_data_raw[-1]
        else:
            pred = test_data_raw[i-1]
        naive_pred.append(pred)
    
    naive_pred = np.array(naive_pred)
    naive_mae = mean_absolute_error(test_data_raw, naive_pred)
    naive_rmse = math.sqrt(mean_squared_error(test_data_raw, naive_pred))
    
    print(f"\nNAIVE Baseline:")
    print(f"  MAE:  {naive_mae:.4f}")
    print(f"  RMSE: {naive_rmse:.4f}")
    
    print(f"\nENSEMBLE:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    if mae < naive_mae:
        improvement = (naive_mae - mae) / naive_mae * 100
        print(f"\n✅ Ensemble is BETTER by {improvement:.1f}%!")
    else:
        decline = (mae - naive_mae) / naive_mae * 100
        print(f"\n⚠️ Naive is BETTER by {decline:.1f}%")
        print("\nPossible reasons:")
        print("  • LSTM overfitting")
        print("  • Features not generalizing")
        print("  • Data too volatile for complex model")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
