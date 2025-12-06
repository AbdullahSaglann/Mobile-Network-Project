import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 60)
print("OUTLIER ANALYSIS - Dataset1")
print("=" * 60)

# Veriyi yükle
df = pd.read_csv("final_processed_dataset1.csv")
print(f"\nToplam Satır Sayısı: {len(df)}")
print(f"Toplam Sütun Sayısı: {len(df.columns)}")

# CellID 1'i seç (kodda kullanılan)
df_cell1 = df[df["CellID"] == 1].copy()
print(f"\nCellID=1 için satır sayısı: {len(df_cell1)}")

# Analiz edilecek kolonlar
numeric_cols = ["smsin", "smsout", "callin", "callout", "internet"]

print("\n" + "=" * 60)
print("1. TEMEL İSTATİSTİKLER")
print("=" * 60)
print(df_cell1[numeric_cols].describe())

print("\n" + "=" * 60)
print("2. OUTLIER TESTİ (IQR Metodu)")
print("=" * 60)

outlier_summary = {}
for col in numeric_cols:
    Q1 = df_cell1[col].quantile(0.25)
    Q3 = df_cell1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_cell1[(df_cell1[col] < lower_bound) | (df_cell1[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df_cell1)) * 100
    
    outlier_summary[col] = {
        'count': outlier_count,
        'percentage': outlier_pct,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'min': df_cell1[col].min(),
        'max': df_cell1[col].max()
    }
    
    print(f"\n{col}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Alt Sınır: {lower_bound:.2f}, Üst Sınır: {upper_bound:.2f}")
    print(f"  Min: {df_cell1[col].min():.2f}, Max: {df_cell1[col].max():.2f}")
    print(f"  Outlier Sayısı: {outlier_count} ({outlier_pct:.2f}%)")

print("\n" + "=" * 60)
print("3. Z-SCORE OUTLIER TESTİ (|Z| > 3)")
print("=" * 60)

for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df_cell1[col]))
    outliers_zscore = df_cell1[z_scores > 3]
    outlier_count = len(outliers_zscore)
    outlier_pct = (outlier_count / len(df_cell1)) * 100
    
    print(f"\n{col}:")
    print(f"  Z-Score Outlier Sayısı: {outlier_count} ({outlier_pct:.2f}%)")

# Görselleştirme
print("\n" + "=" * 60)
print("4. GÖRSELLEŞTİRME OLUŞTURULUYOR...")
print("=" * 60)

# Boxplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df_cell1[col], vert=True)
    axes[idx].set_title(f'{col} - Boxplot')
    axes[idx].set_ylabel('Değer')
    axes[idx].grid(True, alpha=0.3)

# Son subplot'u boş bırak
axes[-1].axis('off')

plt.tight_layout()
plt.savefig("outlier_boxplots.png", dpi=150)
print("✓ Kaydedildi: outlier_boxplots.png")

# Histogram ve dağılım
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df_cell1[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col} - Histogram')
    axes[idx].set_xlabel('Değer')
    axes[idx].set_ylabel('Frekans')
    axes[idx].grid(True, alpha=0.3)

axes[-1].axis('off')

plt.tight_layout()
plt.savefig("outlier_histograms.png", dpi=150)
print("✓ Kaydedildi: outlier_histograms.png")

# Zaman serisi grafikleri
fig, axes = plt.subplots(5, 1, figsize=(15, 12))

df_cell1_sorted = df_cell1.sort_values('datetime').reset_index(drop=True)
df_cell1_sorted['datetime'] = pd.to_datetime(df_cell1_sorted['datetime'])

# İlk 1000 veri noktasını çizelim (daha iyi görünüm için)
sample_size = min(1000, len(df_cell1_sorted))
df_sample = df_cell1_sorted.head(sample_size)

for idx, col in enumerate(numeric_cols):
    axes[idx].plot(df_sample['datetime'], df_sample[col], linewidth=0.5)
    axes[idx].set_title(f'{col} - Zaman Serisi (İlk {sample_size} Nokta)')
    axes[idx].set_ylabel('Değer')
    axes[idx].grid(True, alpha=0.3)
    
    # Outlier boundlarını göster
    if col in outlier_summary:
        upper = outlier_summary[col]['upper_bound']
        axes[idx].axhline(y=upper, color='r', linestyle='--', label=f'Üst Sınır: {upper:.0f}', alpha=0.7)

axes[-1].set_xlabel('Zaman')
plt.tight_layout()
plt.savefig("outlier_timeseries.png", dpi=150)
print("✓ Kaydedildi: outlier_timeseries.png")

print("\n" + "=" * 60)
print("5. ÖNERİLER")
print("=" * 60)

for col in numeric_cols:
    pct = outlier_summary[col]['percentage']
    if pct > 5:
        print(f"\n⚠ {col}: %{pct:.2f} outlier var (yüksek!)")
        print(f"  → Öneri: Bu kolonun outlierlarını cap/clip etmeyi düşünün")
        print(f"  → Üst sınırı {outlier_summary[col]['upper_bound']:.0f} olarak belirleyebilirsiniz")
    elif pct > 1:
        print(f"\n⚡ {col}: %{pct:.2f} outlier var (orta)")
        print(f"  → Öneri: Winsorization veya cap işlemi uygulanabilir")
    else:
        print(f"\n✓ {col}: %{pct:.2f} outlier var (düşük)")

print("\n" + "=" * 60)
print("ANALİZ TAMAMLANDI!")
print("=" * 60)
