"""
Gerçek Ensemble Predictions'ı Yükleyip Baseline ile Karşılaştır
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt

print("=" * 80)
print("🔍 GERÇEK ENSEMBLE PREDICTIONS vs BASELINE")
print("=" * 80)

# Load data
df = pd.read_csv("cleaned_dataset1.csv")
df_cell1 = df[df['CellID'] == 1].copy()
df_cell1['datetime'] = pd.to_datetime(df_cell1['datetime'])
df_cell1 = df_cell1.sort_values('datetime').reset_index(drop=True)

# Feature engineering (same as ensemble)
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

# Split
train_size = int(len(df_cell1) * 0.8)
val_size = int(len(df_cell1) * 0.1)
test_start = train_size + val_size

feat_test = data_feat[test_start:]
target_test = data_target[test_start:]

print(f"\nTest size: {len(target_test)} samples")

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit on training data
feat_train = data_feat[:train_size]
target_train = data_target[:train_size].reshape(-1, 1)

scaler_X.fit(feat_train)
scaler_y.fit(target_train)

# Transform test
X_test_scaled = scaler_X.transform(feat_test)
y_test_scaled = scaler_y.transform(target_test.reshape(-1, 1))

# Create sequences
W_PAST = 48
H_FUTURE = 6

def create_sequences(data_feat, data_target, w_past, h_future):
    X, y = [], []
    for i in range(len(data_feat) - w_past - h_future + 1):
        X.append(data_feat[i : i + w_past])
        y.append(data_target[i + w_past : i + w_past + h_future])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, W_PAST, H_FUTURE)

print(f"Sequence shapes: X_test={X_test.shape}, y_test={y_test.shape}")

# ==========================================
# LOAD ENSEMBLE MODELS
# ==========================================
print("\n" + "=" * 80)
print("📦 LOADING ENSEMBLE MODELS")
print("=" * 80)

try:
    classifier = keras.models.load_model("ensemble_peak_classifier.h5")
    normal_lstm = keras.models.load_model("ensemble_normal_lstm.h5")
    peak_lstm = keras.models.load_model("ensemble_peak_lstm.h5")
    
    print("✓ All 3 models loaded successfully!")
    
    # Get predictions
    print("\nGenerating predictions...")
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
    ensemble_mae = mean_absolute_error(y_test_final, ensemble_final)
    ensemble_mse = mean_squared_error(y_test_final, ensemble_final)
    ensemble_rmse = math.sqrt(ensemble_mse)
    
    mask = y_test_final.flatten() != 0
    ensemble_mape = np.mean(np.abs((y_test_final.flatten()[mask] - ensemble_final.flatten()[mask]) / y_test_final.flatten()[mask])) * 100
    
    print(f"\n✅ ENSEMBLE RESULTS:")
    print(f"  MAE:  {ensemble_mae:.4f}")
    print(f"  RMSE: {ensemble_rmse:.4f}")
    print(f"  MAPE: {ensemble_mape:.2f}%")
    
    ensemble_available = True
    
except Exception as e:
    print(f"❌ Error loading ensemble: {str(e)}")
    ensemble_available = False

# ==========================================
# BASELINE: NAIVE
# ==========================================
print("\n" + "=" * 80)
print("📊 BASELINE: NAIVE FORECAST")
print("=" * 80)

# Use raw test data (not sequences)
test_data_raw = target_test

# Naive predictions
naive_pred = []
for i in range(len(test_data_raw)):
    if i == 0:
        pred = data_target[test_start - 1]  # Last training value
    else:
        pred = test_data_raw[i - 1]
    naive_pred.append(pred)

naive_pred = np.array(naive_pred)

naive_mae = mean_absolute_error(test_data_raw, naive_pred)
naive_rmse = math.sqrt(mean_squared_error(test_data_raw, naive_pred))
mask_naive = test_data_raw != 0
naive_mape = np.mean(np.abs((test_data_raw[mask_naive] - naive_pred[mask_naive]) / test_data_raw[mask_naive])) * 100

print(f"\n✅ NAIVE RESULTS:")
print(f"  MAE:  {naive_mae:.4f}")
print(f"  RMSE: {naive_rmse:.4f}")
print(f"  MAPE: {naive_mape:.2f}%")

# ==========================================
# COMPARISON
# ==========================================
print("\n" + "=" * 80)
print("🔍 DIRECT COMPARISON")
print("=" * 80)

if ensemble_available:
    print(f"\nNaive MAE:    {naive_mae:.4f}")
    print(f"Ensemble MAE: {ensemble_mae:.4f}")
    
    if naive_mae < ensemble_mae:
        diff = ensemble_mae - naive_mae
        pct = (diff / naive_mae) * 100
        print(f"\n⚠️ PROBLEM: Naive is BETTER by {diff:.4f} ({pct:.1f}%)")
        print("\nPossible reasons:")
        print("  1. LSTM overfitting to training data")
        print("  2. Features not helping (noise)")
        print("  3. Scaling issues")
        print("  4. Ensemble complexity hurting generalization")
    else:
        diff = naive_mae - ensemble_mae
        pct = (diff / naive_mae) * 100
        print(f"\n✅ GOOD: Ensemble is BETTER by {diff:.4f} ({pct:.1f}%)")

# Visualization
if ensemble_available:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Predictions comparison (sample)
    N = 500
    sample_true = test_data_raw[:N]
    sample_naive = naive_pred[:N]
    
    # Ensemble needs alignment (it predicts sequences)
    # Take middle forecast from each sequence
    ensemble_simple = ensemble_final[::H_FUTURE].flatten()[:N]
    
    axes[0, 0].plot(sample_true, 'b-', label='Actual', linewidth=2, alpha=0.8)
    axes[0, 0].plot(sample_naive, 'r--', label=f'Naive (MAE: {naive_mae:.2f})', linewidth=1.5)
    axes[0, 0].plot(ensemble_simple, 'g--', label=f'Ensemble (MAE: {ensemble_mae:.2f})', linewidth=1.5)
    axes[0, 0].set_title('Naive vs Ensemble Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE comparison
    axes[0, 1].bar(['Naive', 'Ensemble'], [naive_mae, ensemble_mae], 
                   color=['red', 'green'], edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE Comparison')
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    
    # 3. Error distribution
    ensemble_errors = y_test_final.flatten() - ensemble_final.flatten()
    naive_errors_seq = test_data_raw - naive_pred
    
    axes[1, 0].hist(naive_errors_seq, bins=50, alpha=0.5, label='Naive', color='red', edgecolor='black')
    axes[1, 0].hist(ensemble_errors, bins=50, alpha=0.5, label='Ensemble', color='green', edgecolor='black')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics table
    axes[1, 1].axis('off')
    table_data = [
        ['Metric', 'Naive', 'Ensemble', 'Winner'],
        ['MAE', f'{naive_mae:.2f}', f'{ensemble_mae:.2f}', 'Naive' if naive_mae < ensemble_mae else 'Ensemble'],
        ['RMSE', f'{naive_rmse:.2f}', f'{ensemble_rmse:.2f}', 'Naive' if naive_rmse < ensemble_rmse else 'Ensemble'],
        ['MAPE', f'{naive_mape:.1f}%', f'{ensemble_mape:.1f}%', 'Naive' if naive_mape < ensemble_mape else 'Ensemble']
    ]
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('naive_vs_ensemble_actual.png', dpi=150)
    print("\n✓ Saved: naive_vs_ensemble_actual.png")
    
    plt.show()

print("\n" + "=" * 80)
