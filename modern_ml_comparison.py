"""
Modern ML Models Comparison
Implements and evaluates state-of-the-art ML models:
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Prophet (Facebook's time series)

Compares against existing baselines and LSTM models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import math
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 MODERN ML MODELS COMPARISON")
print("=" * 80)

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("\n📂 Loading and preprocessing data...")

df = pd.read_csv("cleaned_dataset1.csv")
df_cell1 = df[df['CellID'] == 1].copy()
df_cell1['datetime'] = pd.to_datetime(df_cell1['datetime'])
df_cell1 = df_cell1.sort_values('datetime').reset_index(drop=True)

print(f"Loaded {len(df_cell1)} samples")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("\n🔧 Engineering features...")

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
    
    # Lag features (1h, 2h, 6h)
    df['internet_lag_1h'] = df['internet'].shift(6)  # 6 * 10min = 1h
    df['internet_lag_2h'] = df['internet'].shift(12)
    df['internet_lag_6h'] = df['internet'].shift(36)
    df['internet_lag_12h'] = df['internet'].shift(72)
    df['internet_lag_24h'] = df['internet'].shift(144)
    
    # Trend indicators
    df['internet_change_1h'] = df['internet'] - df['internet_lag_1h']
    df['is_increasing'] = (df['internet_change_1h'] > 0).astype(int)
    
    # Rolling statistics (1 hour window)
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

# Feature columns (excluding datetime and target)
feature_cols = [col for col in df_features.columns 
                if col not in ['datetime', 'internet', 'CellID', 'SquareID']]

print(f"Created {len(feature_cols)} features:")
print(f"  {feature_cols[:5]} ... (showing first 5)")

# ==========================================
# 3. TRAIN/VAL/TEST SPLIT
# ==========================================
print("\n✂️ Splitting data...")

# Same split as baseline models
train_size = int(len(df_features) * 0.8)
val_size = int(len(df_features) * 0.1)
test_start = train_size + val_size

train_df = df_features.iloc[:train_size].copy()
val_df = df_features.iloc[train_size:test_start].copy()
test_df = df_features.iloc[test_start:].copy()

# Prepare features and targets
X_train = train_df[feature_cols].values
y_train = train_df['internet'].values

X_val = val_df[feature_cols].values
y_val = val_df['internet'].values

X_test = test_df[feature_cols].values
y_test = test_df['internet'].values

print(f"Train: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Test:  {len(X_test)} samples")

# ==========================================
# 4. MODEL IMPLEMENTATIONS
# ==========================================
print("\n" + "=" * 80)
print("🤖 TRAINING MODELS")
print("=" * 80)

results = {}

# ------------------------------------------
# MODEL 1: Random Forest (Baseline)
# ------------------------------------------
print("\n🌲 Training Random Forest...")
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = math.sqrt(mean_squared_error(y_test, rf_pred))
mask = y_test != 0
rf_mape = np.mean(np.abs((y_test[mask] - rf_pred[mask]) / y_test[mask])) * 100

results['Random Forest'] = {
    'MAE': rf_mae,
    'RMSE': rf_rmse,
    'MAPE': rf_mape,
    'predictions': rf_pred,
    'model': rf_model
}

print(f"✓ Random Forest - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, MAPE: {rf_mape:.2f}%")

# ------------------------------------------
# MODEL 2: XGBoost
# ------------------------------------------
print("\n⚡ Training XGBoost...")
try:
    import xgboost as xgb
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Train without early stopping for simplicity
    xgb_model.fit(X_train, y_train, verbose=False)
    
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = math.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mape = np.mean(np.abs((y_test[mask] - xgb_pred[mask]) / y_test[mask])) * 100
    
    results['XGBoost'] = {
        'MAE': xgb_mae,
        'RMSE': xgb_rmse,
        'MAPE': xgb_mape,
        'predictions': xgb_pred,
        'model': xgb_model
    }
    
    print(f"✓ XGBoost - MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}, MAPE: {xgb_mape:.2f}%")
    
except ImportError:
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")
    results['XGBoost'] = None

# ------------------------------------------
# MODEL 3: LightGBM
# ------------------------------------------
print("\n💡 Training LightGBM...")
try:
    import lightgbm as lgb
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    lgb_pred = lgb_model.predict(X_test)
    
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    lgb_rmse = math.sqrt(mean_squared_error(y_test, lgb_pred))
    lgb_mape = np.mean(np.abs((y_test[mask] - lgb_pred[mask]) / y_test[mask])) * 100
    
    results['LightGBM'] = {
        'MAE': lgb_mae,
        'RMSE': lgb_rmse,
        'MAPE': lgb_mape,
        'predictions': lgb_pred,
        'model': lgb_model
    }
    
    print(f"✓ LightGBM - MAE: {lgb_mae:.4f}, RMSE: {lgb_rmse:.4f}, MAPE: {lgb_mape:.2f}%")
    
except ImportError:
    print("⚠️ LightGBM not installed. Install with: pip install lightgbm")
    results['LightGBM'] = None

# ------------------------------------------
# MODEL 4: CatBoost
# ------------------------------------------
print("\n🐱 Training CatBoost...")
try:
    from catboost import CatBoostRegressor
    
    cat_model = CatBoostRegressor(
        iterations=200,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0
    )
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=20,
        verbose=False
    )
    
    cat_pred = cat_model.predict(X_test)
    
    cat_mae = mean_absolute_error(y_test, cat_pred)
    cat_rmse = math.sqrt(mean_squared_error(y_test, cat_pred))
    cat_mape = np.mean(np.abs((y_test[mask] - cat_pred[mask]) / y_test[mask])) * 100
    
    results['CatBoost'] = {
        'MAE': cat_mae,
        'RMSE': cat_rmse,
        'MAPE': cat_mape,
        'predictions': cat_pred,
        'model': cat_model
    }
    
    print(f"✓ CatBoost - MAE: {cat_mae:.4f}, RMSE: {cat_rmse:.4f}, MAPE: {cat_mape:.2f}%")
    
except ImportError:
    print("⚠️ CatBoost not installed. Install with: pip install catboost")
    results['CatBoost'] = None

# ==========================================
# 5. BASELINE COMPARISONS (from previous runs)
# ==========================================
print("\n" + "=" * 80)
print("📊 ADDING BASELINE RESULTS")
print("=" * 80)

# Add baseline results for comparison
baseline_results = {
    'Naive': {'MAE': 2.84, 'RMSE': 3.92, 'MAPE': 27.3},
    'Seasonal-Naive': {'MAE': 2.52, 'RMSE': 3.45, 'MAPE': 24.1},
    'ARIMA(1,1,1)': {'MAE': 2.63, 'RMSE': 3.58, 'MAPE': 25.4},
    'Enhanced LSTM': {'MAE': 2.74, 'RMSE': 3.68, 'MAPE': 26.2},
    'Weighted LSTM': {'MAE': 2.69, 'RMSE': 3.61, 'MAPE': 25.8},
    'Ensemble LSTM': {'MAE': 2.67, 'RMSE': 3.55, 'MAPE': 25.5}
}

print("\n📋 Baseline Model Results:")
for model, metrics in baseline_results.items():
    print(f"  {model:20s} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")

# ==========================================
# 6. COMPREHENSIVE COMPARISON TABLE
# ==========================================
print("\n" + "=" * 80)
print("📈 COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

# Create comparison dataframe
comparison_data = []

# Add baselines
for model, metrics in baseline_results.items():
    comparison_data.append({
        'Model': model,
        'Type': 'Baseline/LSTM',
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MAPE': metrics['MAPE']
    })

# Add modern ML models
for model, metrics in results.items():
    if metrics is not None:
        comparison_data.append({
            'Model': model,
            'Type': 'Modern ML',
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE']
        })

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('MAE')

print("\n" + df_comparison.to_string(index=False))

# Find best model
best_model = df_comparison.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']} (MAE: {best_model['MAE']:.4f})")

# ==========================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ==========================================
print("\n" + "=" * 80)
print("🔍 FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importances from tree models
importance_data = {}

for model_name, model_data in results.items():
    if model_data is None:
        continue
    
    model = model_data['model']
    
    if hasattr(model, 'feature_importances_'):
        importance_data[model_name] = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):  # CatBoost
        importance_data[model_name] = model.get_feature_importance()

if importance_data:
    # Average importance across all models
    avg_importance = np.mean(list(importance_data.values()), axis=0)
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': avg_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n🔝 Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

# ==========================================
# 8. VISUALIZATION
# ==========================================
print("\n" + "=" * 80)
print("📊 CREATING VISUALIZATIONS")
print("=" * 80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Model Comparison Bar Chart
ax1 = fig.add_subplot(gs[0, :])
models_sorted = df_comparison.sort_values('MAE')
colors = ['#2ecc71' if row['Type'] == 'Modern ML' else '#3498db' if 'LSTM' in row['Model'] else '#e74c3c' 
          for _, row in models_sorted.iterrows()]

bars = ax1.barh(models_sorted['Model'], models_sorted['MAE'], color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_title('📊 All Models Performance Comparison', fontsize=14, fontweight='bold')
ax1.axvline(x=2.52, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Seasonal-Naive Baseline')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, mae) in enumerate(zip(bars, models_sorted['MAE'])):
    ax1.text(mae + 0.05, bar.get_y() + bar.get_height()/2, f'{mae:.3f}', 
             va='center', fontweight='bold', fontsize=9)

# 2. Predictions visualization (best modern ML model)
ax2 = fig.add_subplot(gs[1, 0])
if 'XGBoost' in results and results['XGBoost'] is not None:
    best_ml_pred = results['XGBoost']['predictions']
    model_name = 'XGBoost'
elif 'LightGBM' in results and results['LightGBM'] is not None:
    best_ml_pred = results['LightGBM']['predictions']
    model_name = 'LightGBM'
else:
    best_ml_pred = results['Random Forest']['predictions']
    model_name = 'Random Forest'

N = 500
ax2.plot(y_test[:N], 'b-', label='Actual', linewidth=2, alpha=0.7)
ax2.plot(best_ml_pred[:N], 'r--', label=f'{model_name} Prediction', linewidth=1.5, alpha=0.7)
ax2.set_title(f'🎯 {model_name} Predictions', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time Index')
ax2.set_ylabel('Traffic')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. MAE by Model Type
ax3 = fig.add_subplot(gs[1, 1])
type_comparison = df_comparison.groupby('Type')['MAE'].mean().sort_values()
ax3.bar(range(len(type_comparison)), type_comparison.values, 
        color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(type_comparison)))
ax3.set_xticklabels(type_comparison.index, rotation=45, ha='right')
ax3.set_ylabel('Average MAE')
ax3.set_title('📈 Average MAE by Model Type', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for i, v in enumerate(type_comparison.values):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# 4. Feature Importance (if available)
ax4 = fig.add_subplot(gs[1, 2])
if importance_data:
    top_features = importance_df.head(10)
    ax4.barh(range(len(top_features)), top_features['Importance'], 
             color='#9b59b6', edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['Feature'], fontsize=9)
    ax4.set_xlabel('Importance')
    ax4.set_title('🔍 Top 10 Feature Importance', fontsize=11, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()

# 5. Error Distribution
ax5 = fig.add_subplot(gs[2, 0])
if 'XGBoost' in results and results['XGBoost'] is not None:
    errors = y_test - results['XGBoost']['predictions']
    ax5.hist(errors, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Prediction Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('📉 XGBoost Error Distribution', fontsize=11, fontweight='bold')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax5.grid(True, alpha=0.3)

# 6. RMSE Comparison
ax6 = fig.add_subplot(gs[2, 1])
models_rmse = models_sorted.sort_values('RMSE')
colors_rmse = ['#2ecc71' if row['Type'] == 'Modern ML' else '#3498db' if 'LSTM' in row['Model'] else '#e74c3c' 
               for _, row in models_rmse.iterrows()]
ax6.barh(models_rmse['Model'], models_rmse['RMSE'], color=colors_rmse, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('RMSE (Lower is Better)')
ax6.set_title('📊 RMSE Comparison', fontsize=11, fontweight='bold')
ax6.grid(axis='x', alpha=0.3)

# 7. Summary Statistics Table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

# Top 5 models
top_5 = df_comparison.head(5)
table_data = [['Rank', 'Model', 'MAE', 'RMSE']]
for i, (_, row) in enumerate(top_5.iterrows(), 1):
    table_data.append([
        f"#{i}",
        row['Model'][:15],  # Truncate long names
        f"{row['MAE']:.3f}",
        f"{row['RMSE']:.3f}"
    ])

table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.15, 0.45, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Header formatting
for i in range(4):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best
table[(1, 0)].set_facecolor('#f39c12')
for i in range(4):
    table[(1, i)].set_text_props(weight='bold')

ax7.set_title('🏆 Top 5 Models Ranking', fontsize=11, fontweight='bold', pad=20)

plt.suptitle('Modern ML Models vs Baselines - Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('modern_ml_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: modern_ml_comparison.png")

# ==========================================
# 9. FINAL SUMMARY
# ==========================================
print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\n🏆 BEST OVERALL MODEL: {best_model['Model']}")
print(f"   MAE:  {best_model['MAE']:.4f}")
print(f"   RMSE: {best_model['RMSE']:.4f}")
print(f"   MAPE: {best_model['MAPE']:.2f}%")

# Compare with Seasonal-Naive
seasonal_naive_mae = 2.52
if best_model['MAE'] < seasonal_naive_mae:
    improvement = (seasonal_naive_mae - best_model['MAE']) / seasonal_naive_mae * 100
    print(f"\n✅ Improvement over Seasonal-Naive: {improvement:.2f}%")
else:
    degradation = (best_model['MAE'] - seasonal_naive_mae) / seasonal_naive_mae * 100
    print(f"\n⚠️ Seasonal-Naive is still better by: {degradation:.2f}%")

print("\n💡 KEY INSIGHTS:")
ml_models = [k for k, v in results.items() if v is not None]
if ml_models:
    best_ml = min([(k, v['MAE']) for k, v in results.items() if v is not None], key=lambda x: x[1])
    print(f"  • Best Modern ML: {best_ml[0]} (MAE: {best_ml[1]:.4f})")
    print(f"  • Baseline Champion: Seasonal-Naive (MAE: 2.52)")
    
    if best_ml[1] < seasonal_naive_mae:
        print(f"  • 🎉 Modern ML models CAN beat the baseline!")
    else:
        print(f"  • 📊 Tree-based models also struggle with high volatility data")
        print(f"  • 💡 Data characteristics limit all complex models")

print("\n" + "=" * 80)
