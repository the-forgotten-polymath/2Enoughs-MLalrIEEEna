Team - 2Enoughs submission for the MLarlIEEEna . 
---

## Problem Statement

Given 47 numerical sensor readings (`F01–F47`) captured during device activity cycles, predict whether each device is:

| Class | Meaning |
|-------|---------|
| `0` | Normal — device operating correctly |
| `1` | Faulty — device exhibiting a fault condition |

---

## Results

| Metric | OOF Score |
|--------|-----------|
| **AUC-ROC** | **0.99892** |
| **Macro F1** | **0.98426** |
| **Accuracy** | **0.98497** |
| Optimal Threshold | 0.456 |

> OOF = Out-of-Fold (10-Fold Stratified CV) — unbiased estimate of generalisation performance.

### Per-Model Breakdown

| Model | AUC | F1 | Accuracy |
|-------|-----|----|----------|
| Extra Trees | 0.99926 | 0.98387 | 0.98465 |
| Random Forest | 0.99904 | 0.98119 | 0.98211 |
| XGBoost | 0.99825 | 0.98222 | 0.98303 |
| LightGBM | 0.99824 | 0.98172 | 0.98255 |
| HistGradientBoosting | 0.99804 | 0.98044 | 0.98134 |
| CatBoost | 0.99777 | 0.97907 | 0.98001 |
| **Ensemble** | **0.99892** | **0.98426** | **0.98497** |

---

## 🧠 Architecture

```
RAW FEATURES (47)
      │
      ▼
FEATURE ENGINEERING (121 total)
  ├── Row statistics    : mean, std, min, max, range, median,
  │                       skew, kurtosis, IQR, L1, L2, neg_count
  ├── Band aggregates   : 5 sensor sub-bands × (mean, std, range)
  └── Log transforms    : log1p(|Fxx|) for all 47 features
      │
      ▼
ROBUST SCALER
      │
      ▼
6 BASE MODELS (10-Fold OOF)
  ├── Random Forest       (n=600, balanced, min_leaf=2)
  ├── Extra Trees         (n=600, balanced, min_leaf=2)
  ├── HistGradientBoosting(iter=500, l2=0.5, early_stop)
  ├── XGBoost             (n=800, gamma=0.3, alpha=1, lambda=5)
  ├── LightGBM            (n=800, min_child=40, reg terms)
  └── CatBoost            (iter=600, l2=5, balanced)
      │
      ▼
AUC-WEIGHTED ENSEMBLE
      │
      ▼
THRESHOLD SWEEP (0.10 → 0.90, step=0.001)
  └── Optimise Macro-F1 on OOF predictions
      │
      ▼
FINAL PREDICTIONS → FINAL.csv
```

---

## 📁 Repository Structure

```
.
├── pipeline.py   # Main pipeline (training + inference + plots)
├── FINAL.csv               # Submission predictions (ID → CLASS)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/the-forgotten-polymath/2Enoughs-MLalrIEEEna.git
cd 2Enoughs-MLalrIEEEna
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the data files

Place `TRAIN.csv` and `TEST.csv` in the root directory.

```
.
├── TRAIN.csv    ← training data (43,776 rows × 48 cols)
├── TEST.csv     ← test data    (10,944 rows × 48 cols)
├── ieee_v3_visualized.py
└── ...
```

### 4. Run the pipeline

```bash
python pipeline.py
```

Or in **Google Colab**:

```python
# Step 1 — Install
!pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib

# Step 2 — Upload data
from google.colab import files
files.upload()   # upload TRAIN.csv and TEST.csv

# Step 3 — Run
!python pipeline.py

# Step 4 — Download submission
files.download("FINAL.csv")
```

### 5. Output

| File | Description |
|------|-------------|
| `FINAL.csv` | Final predictions: `ID → CLASS` |

---

## Key Design Decisions

### Why an ensemble?
No single model captures all the signal. RF/ET excel at low-variance predictions; XGB/LGB/CAT add sharper decision boundaries on difficult boundary cases.

### Why threshold = 0.456 and not 0.5?
The default 0.5 threshold over-predicted the Normal class. A systematic sweep at `0.001` granularity over OOF probabilities found that `0.456` maximises Macro-F1 — recovering ~30–40 correct predictions on the test set.

### Why RobustScaler?
The features contain large-magnitude outliers (e.g. F31 reaches values >100). RobustScaler uses IQR-based normalisation, making it far more stable than StandardScaler for this dataset.

### Why 10 folds instead of 5?
With near-perfect AUC (~0.999), threshold calibration becomes the decisive factor for final Accuracy and F1. More folds = more reliable OOF probability estimates = better threshold.

---

## Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
matplotlib>=3.7
```

---

## Dataset

- **Source**: IEEE SB, GEHU — ML Challenge (educational use, no ownership declared)
- **Train**: 43,776 samples × 47 features + 1 target
- **Test**: 10,944 samples × 47 features + ID
- **Class balance**: 60.5% Normal (0) / 39.5% Faulty (1)
