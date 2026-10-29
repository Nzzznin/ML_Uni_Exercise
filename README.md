# Machine Learning — University Exercises

Solutions to the exercises of a university Machine Learning course, written as Jupyter notebooks with Python and scikit-learn. They cover supervised classification, gradient descent implemented from scratch, a stock-market forecasting strategy, and a comparison of clustering algorithms.

## Exercises

### E01 — Comparing classifiers on Iris
[`E01/ML_01.ipynb`](E01/ML_01.ipynb)

Trains six classifiers on the Iris dataset (70/30 split) and compares **accuracy** and **training time** side by side:
Naïve Bayes, K-Nearest Neighbours, Decision Tree, Logistic Regression, Random Forest and an MLP neural network.

### E02 — Gradient descent and market forecasting

**Q1 — Gradient descent from scratch** · [`E02/ML-02-Q1.ipynb`](E02/ML-02-Q1.ipynb)
- Explores and standardises the [Heart Attack dataset](E02/heart.csv) (303 patients, 13 features).
- Implements a `GradientDescentRegressor` with **Batch**, **Stochastic** and **Mini-Batch** gradient descent in NumPy.
- Grid-searches learning rate (0.001 / 0.01 / 0.1) and batch size (16 / 32 / 64), compares MSE, and plots training and test error over iterations to compare convergence.

**Q3 — Market analysis and forecasting** · [`E02/ML_02_Q3.ipynb`](E02/ML_02_Q3.ipynb)
- Downloads 10 years of **S&P 500** prices with `yfinance`.
- Builds technical indicators by hand: 20-day moving average, **RSI** and **MACD**.
- Trains a Random Forest to predict whether the next day's close goes up.
- Back-tests the signal and plots strategy vs. market cumulative return.

### E03 — Clustering algorithms
Six clustering algorithms are applied, scored and visualised (PCA and t-SNE) on two datasets:

- [`E03/ML_03_Fake.ipynb`](E03/ML_03_Fake.ipynb) — a synthetic dataset from `make_blobs` (500 samples, 4 clusters)
- [`E03/ML_03_Kaggle.ipynb`](E03/ML_03_Kaggle.ipynb) — the [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) dataset

| Algorithms | Metrics |
|---|---|
| K-Means, DBSCAN, HDBSCAN, Mean Shift, Agglomerative (hierarchical), Spectral | Silhouette score, Davies–Bouldin index, Calinski–Harabasz index |

## Getting started

```bash
git clone https://github.com/Nzzznin/ML_Uni_Exercise.git
cd ML_Uni_Exercise

pip install numpy pandas matplotlib scikit-learn jupyter yfinance hdbscan ucimlrepo
jupyter lab
```

| Package | Needed for |
|---|---|
| `numpy`, `pandas`, `matplotlib`, `scikit-learn` | all notebooks |
| `yfinance` | E02 Q3 (internet access required) |
| `hdbscan` | E03 |
| `ucimlrepo` | E03 Kaggle notebook (downloads Wine Quality) |

Run the E02 Q1 notebook from inside the `E02/` folder so it can find `heart.csv`.

## Repository structure

```
ML_Uni_Exercise/
├── E01/
│   └── ML_01.ipynb          # Classifier comparison (Iris)
├── E02/
│   ├── ML-02-Q1.ipynb       # Batch / SGD / Mini-batch gradient descent
│   ├── ML_02_Q3.ipynb       # S&P 500 forecasting with RSI & MACD
│   └── heart.csv
└── E03/
    ├── ML_03_Fake.ipynb     # Clustering on synthetic data
    └── ML_03_Kaggle.ipynb   # Clustering on Wine Quality
```
