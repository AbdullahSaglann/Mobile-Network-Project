import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Reshape, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# ==========================================
# 1. SETTINGS
# ==========================================
INPUT_FILE = "cleaned_dataset1.csv"
TARGET_CELL_ID = 1
W_PAST = 48
H_FUTURE = 6
BATCH_SIZE = 64
EPOCHS = 30

tf.random.set_seed(42)
np.random.seed(42)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def create_sequences(data_feat, data_target, w_past, h_future):
    X, y = [], []
    for i in range(len(data_feat) - w_past - h_future + 1):
        X.append(data_feat[i : i + w_past])
        y.append(data_target[i + w_past : i + w_past + h_future])
    return np.array(X), np.array(y)

def build_peak_classifier(input_timesteps, input_features):
    """Binary classifier: Is it a peak?"""
    model = Sequential([
        Input(shape=(input_timesteps, input_features)),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model

def build_lstm_regressor(input_timesteps, input_features, output_timesteps, output_features, name="lstm"):
    """LSTM for traffic prediction"""
    model = Sequential([
        Input(shape=(input_timesteps, input_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(output_timesteps * output_features),
        Reshape((output_timesteps, output_features))
    ], name=name)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# ==========================================
# 3. MAIN
# ==========================================
def main():
    print("=" * 80)
    print("🎭 ENSEMBLE PEAK DETECTION SYSTEM")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(INPUT_FILE)
    df = df[df["CellID"] == TARGET_CELL_ID].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Feature engineering
    print("🎯 Feature engineering...")
    df['hour'] = df['datetime'].dt.hour
    df['is_peak_morning'] = df['hour'].isin([8, 9, 10]).astype(int)
    df['is_peak_evening'] = df['hour'].isin([17, 18, 19, 20]).astype(int)
    df['is_peak'] = (df['is_peak_morning'] | df['is_peak_evening']).astype(int)
    
    df['internet_lag_1h'] = df['internet'].shift(6)
    df['internet_lag_2h'] = df['internet'].shift(12)
    df['internet_lag_6h'] = df['internet'].shift(36)
    df['internet_change_1h'] = df['internet'] - df['internet_lag_1h']
    df['is_increasing'] = (df['internet_change_1h'] > 0).astype(int)
    
    ROLL_WINDOW = 6
    df["internet_roll_mean"] = df["internet"].rolling(ROLL_WINDOW).mean()
    df["internet_roll_std"] = df["internet"].rolling(ROLL_WINDOW).std()
    df["internet_roll_max"] = df["internet"].rolling(ROLL_WINDOW).max()
    df["internet_roll_min"] = df["internet"].rolling(ROLL_WINDOW).min()
    df["internet_roll_75p"] = df["internet"].rolling(ROLL_WINDOW).quantile(0.75)
    df["internet_roll_25p"] = df["internet"].rolling(ROLL_WINDOW).quantile(0.25)
    df["is_weekend"] = (df["datetime"].dt.weekday >= 5).astype(int)
    
    df.fillna(method="bfill", inplace=True)
    df.fillna(0, inplace=True)
    
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
    
    data_feat = df[FEATURE_COLS].values
    data_target = df["internet"].values
    
    # Define peak threshold
    threshold_95 = np.percentile(data_target, 95)
    print(f"\n📊 95th Percentile Threshold: {threshold_95:.2f}")
    
    # Create peak labels
    peak_labels = (data_target >= threshold_95).astype(int)
    print(f"   Normal traffic: {(peak_labels == 0).sum()} ({(peak_labels == 0).sum()/len(peak_labels)*100:.1f}%)")
    print(f"   Peak traffic: {(peak_labels == 1).sum()} ({(peak_labels == 1).sum()/len(peak_labels)*100:.1f}%)")
    
    # Train/val/test split
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.1)
    
    feat_train = data_feat[:train_size]
    feat_val = data_feat[train_size:train_size + val_size]
    feat_test = data_feat[train_size + val_size:]
    
    target_train = data_target[:train_size].reshape(-1, 1)
    target_val = data_target[train_size:train_size + val_size].reshape(-1, 1)
    target_test = data_target[train_size + val_size:].reshape(-1, 1)
    
    peak_train = peak_labels[:train_size]
    peak_val = peak_labels[train_size:train_size + val_size]
    peak_test = peak_labels[train_size + val_size:]
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_raw = scaler_X.fit_transform(feat_train)
    X_val_raw = scaler_X.transform(feat_val)
    X_test_raw = scaler_X.transform(feat_test)
    
    y_train_raw = scaler_y.fit_transform(target_train)
    y_val_raw = scaler_y.transform(target_val)
    y_test_raw = scaler_y.transform(target_test)
    
    # Create sequences
    X_train, y_train = create_sequences(X_train_raw, y_train_raw, W_PAST, H_FUTURE)
    X_val, y_val = create_sequences(X_val_raw, y_val_raw, W_PAST, H_FUTURE)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, W_PAST, H_FUTURE)
    
    # Peak labels for sequences (use middle point of future window)
    peak_train_seq = []
    for i in range(len(X_train)):
        start_idx = i + W_PAST
        peak_train_seq.append(peak_train[start_idx:start_idx + H_FUTURE].mean())
    peak_train_seq = np.array(peak_train_seq)
    
    peak_val_seq = []
    for i in range(len(X_val)):
        start_idx = i + W_PAST  
        peak_val_seq.append(peak_val[start_idx:start_idx + H_FUTURE].mean())
    peak_val_seq = np.array(peak_val_seq)
    
    peak_test_seq = []
    for i in range(len(X_test)):
        start_idx = i + W_PAST
        peak_test_seq.append(peak_test[start_idx:start_idx + H_FUTURE].mean())
    peak_test_seq = np.array(peak_test_seq)
    
    print(f"\n📐 Sequence shapes:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   Peak labels: {peak_train_seq.shape}")
    
    # ==========================================
    # STEP 1: Train Peak Classifier
    # ==========================================
    print("\n" + "=" * 80)
    print("🎯 STEP 1: Training Peak Classifier")
    print("=" * 80)
    
    classifier = build_peak_classifier(W_PAST, X_train.shape[2])
    classifier.summary()
    
    # Class weights (peaks are rare!)
    class_weight = {0: 1.0, 1: 19.0}  # 95:5 ratio
    
    history_clf = classifier.fit(
        X_train, peak_train_seq,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, peak_val_seq),
        class_weight=class_weight,
        verbose=2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
    )
    
    classifier.save("ensemble_peak_classifier.h5")
    print("\n✓ Classifier saved: ensemble_peak_classifier.h5")
    
    # ==========================================
    # STEP 2: Train Normal Traffic LSTM
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 STEP 2: Training Normal Traffic LSTM")
    print("=" * 80)
    
    normal_lstm = build_lstm_regressor(W_PAST, X_train.shape[2], H_FUTURE, 1, "normal_lstm")
    
    history_normal = normal_lstm.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
    )
    
    normal_lstm.save("ensemble_normal_lstm.h5")
    print("\n✓ Normal LSTM saved: ensemble_normal_lstm.h5")
    
    # ==========================================
    # STEP 3: Train Peak Traffic LSTM (with oversampling)
    # ==========================================
    print("\n" + "=" * 80)
    print("⚡ STEP 3: Training Peak Traffic LSTM (Specialized)")
    print("=" * 80)
    
    # Oversample peaks in training data (repeat 5x)
    peak_indices = np.where(peak_train_seq > 0.5)[0]
    oversampled_indices = np.concatenate([
        np.arange(len(X_train)),
        np.tile(peak_indices, 5)  # Repeat peaks 5 times
    ])
    
    X_train_over = X_train[oversampled_indices]
    y_train_over = y_train[oversampled_indices]
    
    print(f"   Original training size: {len(X_train)}")
    print(f"   Oversampled training size: {len(X_train_over)}")
    
    peak_lstm = build_lstm_regressor(W_PAST, X_train.shape[2], H_FUTURE, 1, "peak_lstm")
    
    history_peak = peak_lstm.fit(
        X_train_over, y_train_over,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
    )
    
    peak_lstm.save("ensemble_peak_lstm.h5")
    print("\n✓ Peak LSTM saved: ensemble_peak_lstm.h5")
    
    # ===========================================
    # STEP 4: Ensemble Prediction
    # ==========================================
    print("\n" + "=" * 80)
    print("🎭 STEP 4: Ensemble Prediction")
    print("=" * 80)
    
    # Get predictions
    peak_prob = classifier.predict(X_test, verbose=0).flatten()
    normal_pred = normal_lstm.predict(X_test, verbose=0)
    peak_pred = peak_lstm.predict(X_test, verbose=0)
    
    # Smart combination
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
    
    print("\n" + "=" * 80)
    print("📈 ENSEMBLE RESULTS")
    print("=" * 80)
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print("=" * 80)
    
    # Visualization
    N = 500
    subset_true = y_test_final[-N:]
    subset_pred = ensemble_final[-N:]
    
    plt.figure(figsize=(16, 6))
    plt.plot(subset_true, "b-", label="Gerçek", linewidth=2, alpha=0.8)
    plt.plot(subset_pred, "r--", label="Ensemble Prediction", linewidth=2, alpha=0.8)
    plt.title(f"🎭 ENSEMBLE SYSTEM (MAE: {mae:.2f}, RMSE: {rmse:.2f})", fontsize=14, fontweight='bold')
    plt.xlabel("Zaman")
    plt.ylabel("Internet Trafiği")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ensemble_predictions.png", dpi=150)
    print("\n✓ Saved: ensemble_predictions.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ ENSEMBLE SYSTEM COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
