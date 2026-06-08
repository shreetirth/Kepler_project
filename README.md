# 🪐 Kepler Exoplanet Search — ML Pipeline

A multi-stage machine learning pipeline built on NASA's Kepler Exoplanet Search Results dataset. The project goes beyond standard classification by chaining regression, clustering, classification, and validation into a coherent end-to-end analysis.
---

## 📋 Dataset

**Source:** [NASA Kepler Exoplanet Search Results — Kaggle](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)

- ~9,564 rows, 50 columns
- Cumulative record of all Kepler Objects of Interest (KOIs)
- Target variable: `koi_disposition` — CONFIRMED, FALSE POSITIVE, CANDIDATE

---

## 🧠 Pipeline Overview

| Step | Method | Target | Result |
|---|---|---|---|
| 1 | Multiple Linear Regression | Predict `koi_score` | R²=0.755, RMSE=0.2 |
| 2 | KMeans Clustering (k=2) | Cluster by score | Natural separation confirmed |
| 3 | Random Forest Classifier | Predict `koi_disposition` | 75.6% accuracy |
| 4 | Confusion Matrix | Automated vs scientific verdict | 44 confirmed planets nearly missed |

---

## 🔍 Key Findings

- **False positive flags alone** explain 75.5% of the variance in `koi_score` — confirming they are strong predictors of Kepler's confidence
- **KMeans with k=2** on `koi_score` naturally separated CANDIDATEs from FALSE POSITIVEs without using any labels — validating the score as a meaningful signal
- **`koi_prad`** (planetary radius) was the most important feature in the Random Forest, followed by `koi_depth` and `koi_impact` — physical properties matter more than stellar properties for classification
- **44 confirmed planets** were flagged as FALSE POSITIVE by Kepler's automated pipeline but later confirmed by scientific review — the most significant finding of this project

---

## 📊 Visualizations

- Class balance countplot
- Correlation heatmap of numerical features
- Log-scale histplots for `koi_period`, `koi_duration`, `koi_depth`
- Pairplot of top features colored by disposition
- Boxplot of `koi_score` by disposition
- Elbow method plot for optimal k
- Feature importance bar chart
- Confusion matrix heatmap

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Machine Learning** — scikit-learn
  - `LinearRegression`
  - `KMeans`
  - `RandomForestClassifier`
  - `confusion_matrix`, `classification_report`

---

## 📁 Project Structure

```
kepler-project/
│
├── kepler-project.ipynb    # Main notebook with full pipeline
└── README.md               # This file
```

---

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)
2. Open `kepler-project.ipynb` in Kaggle or Jupyter
3. Update the `file_path` in Cell 3 to point to your local CSV
4. Run all cells in order

---

## 💡 What I Learned

- How to identify scale issues from `describe()` output and apply log scale fixes
- Why blindly dropping nulls can wipe out an entire dataset — and how to use `subset` and column thresholds instead
- How KMeans can validate a scoring system without using labels
- Why the CANDIDATE class is inherently harder to classify — it sits in the uncertain middle ground between confirmed and rejected
- The importance of human scientific review on top of automated ML pipelines — 44 real planets were nearly dismissed

---

## 📌 Known Limitations

- CANDIDATE class F1 score is 0.49 — inherent uncertainty in the class makes it hard to classify
- `koi_score` was excluded from Random Forest features to avoid data leakage
- Model accuracy could potentially improve with feature engineering or hyperparameter tuning

---
