# BetaBytez AI/ML Internship — Task 2
### Model Training on Two Datasets with Comparison

## 📊 Datasets

| Dataset | Domain | Samples | Features | Classes |
|---|---|---|---|---|
| **Wine Recognition** | Chemical / Agricultural | 178 | 13 | 3 (cultivars) |
| **Breast Cancer Wisconsin** | Medical | 569 | 30 | 2 (malignant/benign) |

Both datasets are bundled directly with scikit-learn (`load_wine`, `load_breast_cancer`), so they load
reliably with no external download and no missing/corrupted values — letting the analysis focus fully on
modeling and comparison rather than data cleaning.

## 🧠 Approach (per dataset)

1. **EDA** — class balance, feature distributions by class, and a correlation heatmap (3 plots per dataset,
   6 total).
2. **Preprocessing** — stratified 80/20 train/test split, `StandardScaler` for the linear model.
3. **Modeling** — trained Logistic Regression and Random Forest on each dataset.
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, and a confusion matrix per model.
5. **Improvement step** — used `GridSearchCV` to tune the hyperparameters of the weaker model on each
   dataset (Logistic Regression's `C` on Wine; Random Forest's depth/leaf-size on Breast Cancer).
6. **Comparative Analysis Report** — written directly in the notebook (markdown cells), answering which
   dataset was harder, which model generalized better, and what this shows about data quality vs.
   performance.

## 📈 Results

**Wine (multiclass, weighted average metrics):**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 97.2% | 97.4% | 97.2% | 97.2% |
| **Random Forest** ✅ | **100%** | **100%** | **100%** | **100%** |

**Breast Cancer (binary):**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **98.2%** | **98.6%** | **98.6%** | **98.6%** |
| Random Forest | 95.6% | 95.9% | 97.2% | 96.6% |

**Key finding:** No single model won on both datasets — Random Forest excelled on Wine's non-linear
chemical feature interactions, while Logistic Regression excelled on Breast Cancer's more linearly
separable feature space. GridSearchCV tuning of the weaker model in each case produced only marginal
change, showing both models were already close to their performance ceiling on these clean, well-prepared
datasets — the full reasoning is written out in the notebook's Comparative Analysis Report section.

## 📁 Repository Structure

```
betabytez-aiml-task2-yourname/
├── task2_notebook.ipynb     # Full EDA, preprocessing, training, evaluation, tuning, and comparison report
├── requirements.txt
└── README.md
```

## ▶️ How to Run

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook task2_notebook.ipynb
```
Then run all cells (**Cell → Run All**) to reproduce the full EDA, training, evaluation, and comparison report.

---
BetaBytez Summer Internship 2026 — AI/ML Track — Task 2
