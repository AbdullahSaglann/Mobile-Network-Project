import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("OUTLIER TEMİZLEME - Winsorization Yöntemi")
print("=" * 70)

# Veriyi yükle
INPUT_FILE = "final_processed_dataset1.csv"
OUTPUT_FILE = "cleaned_dataset1.csv"

print(f"\nGiriş Dosyası: {INPUT_FILE}")
print(f"Çıkış Dosyası: {OUTPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print(f"\nToplam Satır: {len(df)}")
print(f"Toplam CellID Sayısı: {df['CellID'].nunique()}")

# Temizleme öncesi istatistikler
numeric_cols = ["smsin", "smsout", "callin", "callout", "internet"]

print("\n" + "=" * 70)
print("TEMİZLEME ÖNCESİ İSTATİSTİKLER")
print("=" * 70)
print(df[numeric_cols].describe())

def cap_outliers_iqr(df, column, multiplier=1.5):
    """
    IQR yöntemiyle outlierları cap eder (winsorization)
    multiplier: 1.5 (standart) veya 3.0 (daha toleranslı)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Outlierları sınırlarla cap et
    original_min = df[column].min()
    original_max = df[column].max()
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    capped_count = ((df[column] == lower_bound) | (df[column] == upper_bound)).sum()
    
    return {
        'column': column,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'original_min': original_min,
        'original_max': original_max,
        'capped_count': capped_count
    }

# Her CellID için ayrı ayrı temizle (her hücrenin farklı trafik özellikleri olabilir)
df_cleaned = pd.DataFrame()

for cell_id in df['CellID'].unique():
    print(f"\n--- CellID: {cell_id} İşleniyor ---")
    df_cell = df[df['CellID'] == cell_id].copy()
    
    for col in numeric_cols:
        stats = cap_outliers_iqr(df_cell, col, multiplier=1.5)
        
        if stats['capped_count'] > 0:
            print(f"  {col}:")
            print(f"    Orijinal Aralık: [{stats['original_min']:.2f}, {stats['original_max']:.2f}]")
            print(f"    Yeni Sınırlar: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
            print(f"    Cap Edilen Değer: {stats['capped_count']}")
    
    df_cleaned = pd.concat([df_cleaned, df_cell], ignore_index=True)

print("\n" + "=" * 70)
print("TEMİZLEME SONRASI İSTATİSTİKLER")
print("=" * 70)
print(df_cleaned[numeric_cols].describe())

# Karşılaştırma grafiği (CellID 1 için)
df_cell1_before = df[df['CellID'] == 1].copy()
df_cell1_after = df_cleaned[df_cleaned['CellID'] == 1].copy()

fig, axes = plt.subplots(5, 2, figsize=(14, 18))

for idx, col in enumerate(numeric_cols):
    # Önce
    axes[idx, 0].hist(df_cell1_before[col], bins=50, edgecolor='black', alpha=0.7, color='red')
    axes[idx, 0].set_title(f'{col} - Önce (Outlierlar ile)')
    axes[idx, 0].set_ylabel('Frekans')
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Sonra
    axes[idx, 1].hist(df_cell1_after[col], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[idx, 1].set_title(f'{col} - Sonra (Temiz)')
    axes[idx, 1].set_ylabel('Frekans')
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("before_after_cleaning.png", dpi=150)
print("\n✓ Kaydedildi: before_after_cleaning.png")

# Temizlenmiş veriyi kaydet
df_cleaned.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ TEMİZLENMİŞ VERİ KAYDEDİLDİ: {OUTPUT_FILE}")

print("\n" + "=" * 70)
print("ÖNEMLİ NOT:")
print("=" * 70)
print("1. Outlierlar KALDIRILMADI, CAP EDİLDİ (Winsorization)")
print("2. Bu yöntem veri kaybını önler ve daha stabil eğitim sağlar")
print("3. Artık lstm_real.py'de INPUT_FILE'ı 'cleaned_dataset1.csv' yapın")
print("=" * 70)

# Dosya boyutu karşılaştırması
import os
original_size = os.path.getsize(INPUT_FILE) / (1024**3)  # GB
cleaned_size = os.path.getsize(OUTPUT_FILE) / (1024**3)  # GB

print(f"\nOrijinal Dosya Boyutu: {original_size:.2f} GB")
print(f"Temizlenmiş Dosya Boyutu: {cleaned_size:.2f} GB")
print(f"Satır Sayısı (Değişmedi): {len(df_cleaned)}")
