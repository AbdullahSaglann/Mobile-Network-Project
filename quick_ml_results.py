"""
Quick script to extract just the final results from modern ML comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import math
import warnings
import pickle
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("cleaned_dataset1.csv")
df_cell1 = df[df['CellID'] == 1].copy()
df_cell1['datetime'] = pd.to_datetime(df_cell1['datetime'])
df_cell1 = df_cell1.sort_values('datetime').reset_index(drop=True)

def create_features(df):
    """Create comprehensive features for ML models"""
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['minute'] = df['datetime'].dt.minute
    
    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Peak hour indicators
    df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features
    df['internet_lag_1h'] = df['internet'].shift(6)
    df['internet_lag_2h'] = df['internet'].shift(12)
    df['internet_lag_6h'] = df['internet'].shift(36)
    df['internet_lag_12h'] = df['internet'].shift(72)
    df['internet_lag_24h'] = df['internet'].shift(144)
    
    # Trend indicators
    df['internet_change_1h'] = df['internet'] - df['internet_lag_1h']
    df['is_increasing'] = (df['internet_change_1h'] > 0).astype(int)
    
    # Rolling statistics
    window = 6
    df['rolling_mean'] = df['internet'].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df['internet'].rolling(window=window, min_periods=1).std()
    df['rolling_max'] = df['internet'].rolling(window=window, min_periods=1).max()
    df['rolling_min'] = df['internet'].rolling(window=window, min_periods=1).min()
    df['rolling_q75'] = df['internet'].rolling(window=window, min_periods=1).quantile(0.75)
    df['rolling_q25'] = df['internet'].rolling(window=window, min_periods=1).quantile(0.25)
    
    # SMS and Call features (if available)
    if 'smsin' in df.columns:
        df['smsin_lag_1h'] = df['smsin'].shift(6)
        df['smsout_lag_1h'] = df['smsout'].shift(6)
        df['callin_lag_1h'] = df['callin'].shift(6)
        df['callout_lag_1h'] = df['callout'].shift(6)
    
    # Fill NaN values
    df = df.bfill().ffill()
    
    return df

df_features = create_features(df_cell1)

# Feature columns
feature_cols = [col for col in df_features.columns 
                if col not in ['datetime', 'internet', 'CellID', 'SquareID']]

# Split data
train_size = int(len(df_features) * 0.8)
val_size = int(len(df_features) * 0.1)
test_start = train_size + val_size

train_df = df_features.iloc[:train_size].copy()
test_df = df_features.iloc[test_start:].copy()

X_train = train_df[feature_cols].values
y_train = train_df['internet'].values
X_test = test_df[feature_cols].values
y_test = test_df['internet'].values

print("="*80)
print("🚀 MODERN ML MODELS - FINAL RESULTS")
print("="*80)

results = {}

# Random Forest
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=0)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = math.sqrt(mean_squared_error(y_test, rf_pred))
mask = y_test != 0
rf_mape = np.mean(np.abs((y_test[mask] - rf_pred[mask]) / y_test[mask])) * 100
results['Random Forest'] = {'MAE': rf_mae, 'RMSE': rf_rmse, 'MAPE': rf_mape}
print(f"   MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, MAPE: {rf_mape:.2f}%")

# XGBoost
try:
    print("\n⚡ Training XGBoost...")
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, 
                                  random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = math.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mape = np.mean(np.abs((y_test[mask] - xgb_pred[mask]) / y_test[mask])) * 100
    results['XGBoost'] = {'MAE': xgb_mae, 'RMSE': xgb_rmse, 'MAPE': xgb_mape}
    print(f"   MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}, MAPE: {xgb_mape:.2f}%")
except:
    print("   ⚠️ XGBoost failed")

# LightGBM
try:
    print("\n💡 Training LightGBM...")
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, 
                                   random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    lgb_rmse = math.sqrt(mean_squared_error(y_test, lgb_pred))
    lgb_mape = np.mean(np.abs((y_test[mask] - lgb_pred[mask]) / y_test[mask])) * 100
    results['LightGBM'] = {'MAE': lgb_mae, 'RMSE': lgb_rmse, 'MAPE': lgb_mape}
    print(f"   MAE: {lgb_mae:.4f}, RMSE: {lgb_rmse:.4f}, MAPE: {lgb_mape:.2f}%")
except:
    print("   ⚠️ LightGBM failed")

# CatBoost
try:
    print("\n🐱 Training CatBoost...")
    from catboost import CatBoostRegressor
    cat_model = CatBoostRegressor(iterations=200, depth=8, learning_rate=0.05, 
                                   random_seed=42, verbose=0)
    cat_model.fit(X_train, y_train, verbose=False)
    cat_pred = cat_model.predict(X_test)
    cat_mae = mean_absolute_error(y_test, cat_pred)
    cat_rmse = math.sqrt(mean_squared_error(y_test, cat_pred))
    cat_mape = np.mean(np.abs((y_test[mask] - cat_pred[mask]) / y_test[mask])) * 100
    results['CatBoost'] = {'MAE': cat_mae, 'RMSE': cat_rmse, 'MAPE': cat_mape}
    print(f"   MAE: {cat_mae:.4f}, RMSE: {cat_rmse:.4f}, MAPE: {cat_mape:.2f}%")
except:
    print("   ⚠️ CatBoost failed")

# Comparison
print("\n" + "="*80)
print("📊 COMPREHENSIVE COMPARISON")
print("="*80)

baseline_results = {
    'Naive': {'MAE': 2.84, 'RMSE': 3.92, 'MAPE': 27.3},
    'Seasonal-Naive': {'MAE': 2.52, 'RMSE': 3.45, 'MAPE': 24.1},
    'ARIMA': {'MAE': 2.63, 'RMSE': 3.58, 'MAPE': 25.4},
    'Enhanced LSTM': {'MAE': 2.74, 'RMSE': 3.68, 'MAPE': 26.2},
    'Weighted LSTM': {'MAE': 2.69, 'RMSE': 3.61, 'MAPE': 25.8},
    'Ensemble LSTM': {'MAE': 2.67, 'RMSE': 3.55, 'MAPE': 25.5}
}

all_results = {**baseline_results, **results}

print(f"\n{'Model':<20} {'Type':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
print("-"*80)

sorted_models = sorted(all_results.items(), key=lambda x: x[1]['MAE'])
for model, metrics in sorted_models:
    model_type = 'Modern ML' if model in results else 'Baseline/LSTM'
    print(f"{model:<20} {model_type:<15} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['MAPE']:<10.2f}")

print("\n" + "="*80)
best = sorted_models[0]
print(f"🏆 BEST MODEL: {best[0]}")
print(f"   MAE: {best[1]['MAE']:.4f}")
print(f"   RMSE: {best[1]['RMSE']:.4f}")
print(f"   MAPE: {best[1]['MAPE']:.2f}%")

if best[1]['MAE'] < 2.52:
    improvement = (2.52 - best[1]['MAE']) / 2.52 * 100
    print(f"\n✅ {improvement:.2f}% improvement over Seasonal-Naive baseline!")
else:
    degradation = (best[1]['MAE'] - 2.52) / 2.52 * 100
    print(f"\n⚠️ Seasonal-Naive baseline still {degradation:.2f}% better")

print("\n💡 KEY FINDINGS:")
ml_best = min([(k, v['MAE']) for k, v in results.items()], key=lambda x: x[1])
print(f"  • Best Modern ML: {ml_best[0]} (MAE: {ml_best[1]:.4f})")
print(f"  • vs Seasonal-Naive: {ml_best[1]:.4f} vs 2.52")

if ml_best[1] < 2.52:
    print(f"  • 🎉 Success! Modern ML beats simple baseline")
else:
    pct_worse = (ml_best[1] - 2.52) / 2.52 * 100
    print(f"  • 📊 Modern ML {pct_worse:.1f}% worse than simple baseline")
    print(f"  • 💡 Confirms: Data volatility limits ALL complex models")

print("\n" + "="*80)
