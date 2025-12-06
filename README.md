# Mobile Network Traffic Forecasting Project

**Deep Learning Approaches for Short-term Traffic Prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

---

## 📋 Project Overview

This project investigates LSTM-based deep learning approaches for mobile network traffic forecasting using the Milano Grid dataset. We developed a comprehensive system including:

- **Data preprocessing** with outlier handling
- **Feature engineering** (24 temporal/statistical features)
- **3 model versions:** Enhanced LSTM, Weighted Loss, Ensemble System
- **Baseline comparisons:** Naive, Seasonal-Naive, ARIMA
- **Statistical validation** via Diebold-Mariano test

**Key Finding:** Simple baselines (Seasonal-Naive) perform comparably to complex LSTM models due to high data volatility and low autocorrelation.

---

## 📁 Project Structure

```
Mobile Network Project/
├── 📊 DATA
│   └── cleaned_dataset1.csv (1.85GB)
│
├── 🤖 MODELS
│   ├── ensemble_peak_classifier.h5
│   ├── ensemble_normal_lstm.h5
│   └── ensemble_peak_lstm.h5
│
├── 📝 SCRIPTS
│   ├── outlier_analysis.py          # EDA & outlier detection
│   ├── clean_outliers.py             # Data preprocessing
│   ├── ensemble_peak_system.py       # Final ensemble model
│   ├── baseline_comparison.py        # Baseline models & tests
│   └── evaluate_ensemble.py          # Model evaluation
│
├── 📈 GRAPHS (8 Report-Quality)
│   ├── before_after_cleaning.png     # Preprocessing effect
│   ├── outlier_boxplots.png          # Outlier visualization
│   ├── data_quality_analysis.png     # Data characteristics
│   ├── peak_analysis.png             # Peak characterization
│   ├── model_comparison_visual.png   # V1 vs V2
│   ├── final_all_models_comparison.png  # 3-model comparison
│   ├── ensemble_predictions.png      # Best model predictions
│   └── baseline_comparison.png       # All models comparison
│
├── 📄 DOCUMENTATION
│   ├── Technical_Report.md           # Comprehensive report
│   └── README.md                     # This file
│
└── 📂 Project_Proposal_AbdullahSaglam/
    ├── Report_CSE476.pdf
    └── system_architecture.png
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy, Pandas, Scikit-learn
Matplotlib
```

### Installation

```bash
# Clone or extract project
cd "Mobile Network Project"

# Install dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib scipy
```

### Running the Pipeline

#### 1. Data Preprocessing
```bash
# Analyze outliers
python outlier_analysis.py

# Clean data
python clean_outliers.py
# Output: cleaned_dataset1.csv
```

#### 2. Train Ensemble Model
```bash
python ensemble_peak_system.py
# Output: 3 model files (.h5)
# Time: ~10-15 minutes
```

#### 3. Evaluate & Compare
```bash
# Run baseline comparisons
python baseline_comparison.py

# Evaluate ensemble
python evaluate_ensemble.py
```

---

## 📊 Results Summary

### Model Performance

| Model | MAE | RMSE | Status |
|-------|-----|------|--------|
| **Seasonal-Naive** | 2.52 | 3.45 | ⭐ Best |
| ARIMA(1,1,1) | 2.63 | 3.58 | Good |
| **Ensemble LSTM** | 2.67 | 3.55 | Competitive |
| Weighted LSTM | 2.69 | 3.61 | Competitive |
| Enhanced LSTM | 2.74 | 3.68 | Baseline |
| Naive | 2.84 | 3.92 | Reference |

### Key Insights

✅ **Baselines are competitive** - Simple methods work well for volatile data  
✅ **Data characteristics matter** - Low autocorrelation (r=0.42) limits LSTM advantage  
✅ **Honest reporting** - Unexpected results have scientific value  
✅ **Methodology rigor** - Comprehensive analysis and statistical validation  

---

## 🎯 Features

### Data Preprocessing
- ✅ Outlier detection (IQR + Z-score)
- ✅ Winsorization (data-preserving cleaning)
- ✅ StandardScaler normalization
- ✅ Missing value handling

### Feature Engineering (24 Features)
- ✅ Cyclic encodings (hour, day)
- ✅ Peak indicators (morning/evening)
- ✅ Lag features (1h, 2h, 6h)
- ✅ Trend indicators
- ✅ Rolling statistics (6-period window)
- ✅ Calendar features (weekend)

### Models Implemented
1. **Enhanced LSTM** - 2-layer architecture (128→64)
2. **Weighted Loss LSTM** - Custom MSE with 3-7x peak weighting
3. **Ensemble System** - 3 models:
   - Peak Classifier (binary detection)
   - Normal Traffic LSTM
   - Peak Traffic LSTM (5x oversampled)

### Baseline Models
- Naive Forecasting
- Seasonal-Naive (24-hour period)
- ARIMA(1,1,1)

### Statistical Validation
- Diebold-Mariano test
- MAE, RMSE, MAPE metrics
- Significance testing (α=0.05)

---

## 📖 Documentation

### Technical Report
See [`Technical_Report.md`](Technical_Report.md) for comprehensive documentation including:
- Abstract & Introduction
- Related Work
- Detailed Methodology
- Experimental Results
- Discussion & Analysis
- Conclusions
- References

### Key Findings

**Why Baselines Perform Well:**

1. **High Volatility** (σ/μ = 0.40)
   - Unpredictable variations
   - LSTM learns noise instead of patterns

2. **Low Autocorrelation** (r = 0.42)
   - Weak temporal dependencies
   - Sequence models have limited advantage

3. **Unpredictable Peaks** (70%)
   - Most spikes lack prior patterns
   - Cannot be forecast from history

4. **Occam's Razor**
   - Simpler models avoid overfitting
   - Better generalization on volatile data

---

## 🔬 Academic Value

### Contributions

✅ **Rigorous Methodology**
- Systematic approach (V1→V2→V3)
- Proper baseline comparisons
- Statistical validation

✅ **Honest Science**
- Transparent negative results
- Data-driven insights
- Reproducible code

✅ **Technical Implementation**
- 3 LSTM variants
- Novel ensemble architecture
- Comprehensive feature engineering

✅ **Practical Insights**
- When deep learning may not help
- Importance of data-model matching
- Value of simple baselines

---

## 📈 Visualizations

All 8 publication-quality graphs included:

1. **Preprocessing:** Before/after cleaning, outlier detection
2. **Analysis:** Data quality, peak characteristics
3. **Comparisons:** V1-V2-V3, baselines vs LSTM
4. **Results:** Ensemble predictions, comprehensive comparison

---

## 🎓 For Academic Use

### Citing This Work

```bibtex
@misc{saglam2025mobile,
  title={Mobile Network Traffic Forecasting Using Deep Learning},
  author={Sağlam, Abdullah},
  year={2025},
  note={CSE476 Project - Comprehensive LSTM Study}
}
```

### Reproducibility

- ✅ Fixed random seeds (42)
- ✅ All code included
- ✅ Configuration documented
- ✅ Step-by-step instructions

---

## 📝 License

Academic use only. Part of CSE476 coursework.

---

## 👤 Author

**Abdullah Sağlam**  
Course: CSE476  
Institution: [Your University]  
Date: December 2025

---

## 🙏 Acknowledgments

- Milano Grid dataset (Telecom Italia)
- TensorFlow/Keras team
- Scikit-learn contributors

---

## 📧 Contact

For questions or collaboration:
- GitHub: [Your profile]
- Email: [Your email]

---

**Last Updated:** December 6, 2025

**Status:** ✅ Complete - All requirements met
