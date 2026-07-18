# Rough-set-Machine-Learning_project
Predictive Modeling of Biochar Adsorption Capacity using Variable Precision Rough Set Machine Learning (VPRS-ML)
# FC-VPRS-ML
## Fuzzy-Conformal Variable Precision Rough Set Machine Learning for Hydrochar Adsorption Capacity Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project presents an enhanced **Variable Precision Rough Set Machine Learning (VPRS-ML)** framework for predicting hydrochar adsorption capacity. Instead of treating the model as a black box, the proposed approach focuses on **interpretability, uncertainty quantification, and robust evaluation**, making it suitable for scientific and environmental engineering applications.

The framework integrates Rough Set Theory, Fuzzy Logic, and Conformal Prediction into a unified classification model capable of generating physically interpretable decision rules while maintaining competitive predictive performance.

---

# Motivation

Traditional VPRS classifiers suffer from several limitations:

- Fixed manually selected precision parameter (β)
- Crisp decision boundaries
- No uncertainty estimation
- Limited evaluation metrics
- Poor interpretability after feature scaling
- High variance from single train-test splits

This project addresses these issues through six methodological improvements.

---

# Proposed Methodology

## 1. Adaptive β Selection

Instead of manually selecting the Variable Precision parameter (β), the model automatically determines the optimal value through internal stratified validation.

**Benefits**

- Eliminates manual tuning
- Improves reproducibility
- Reduces bias
- Makes the framework self-calibrating

---

## 2. Fuzzy Rule Inference

Classical rough set rules are retained during rule induction.

During prediction, crisp rule matching is replaced with triangular fuzzy membership functions.

Benefits:

- Smooth decision boundaries
- Better handling of boundary samples
- Eliminates hard threshold effects
- Removes ad-hoc fallback prediction

---

## 3. Split Conformal Prediction

The framework produces prediction sets rather than only a single predicted class.

Example:

Instead of

High-Q

the model may output

{Medium-Q, High-Q}

when uncertainty exists.

Advantages:

- Finite-sample coverage guarantee
- Reliable uncertainty estimation
- Better scientific interpretability

---

## 4. Ordinal-aware Evaluation

Since adsorption capacity classes are ordered

Low → Medium → High

the framework evaluates performance using

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Quadratic Weighted Kappa (QWK)
- Ordinal Mean Absolute Error (MAE)

These metrics better reflect prediction quality than accuracy alone.

---

## 5. Robust Benchmarking

Instead of relying on a single train-test split, the framework performs

Repeated Stratified K-Fold Cross Validation

and compares performance against

- Classical VPRS
- Decision Tree
- Random Forest

This provides statistically reliable model evaluation.

---

## 6. Physically Interpretable Rule Extraction

Unlike many ML models that generate abstract feature importance scores, this framework exports decision rules directly in original measurement units.

Example:

IF C0(mg/g) ∈ [0.08,17.9]

THEN

Low-Q

These rules can be interpreted directly by researchers and domain experts.

---

# Workflow

Dataset
↓

Data Cleaning

↓

Target Discretization

↓

Feature Discretization

↓

Adaptive β Selection

↓

VPRS Rule Induction

↓

Fuzzy Rule Matching

↓

Conformal Prediction

↓

Evaluation

↓

Decision Rule Export

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Rough Set Theory
- Variable Precision Rough Sets (VPRS)
- Fuzzy Logic
- Split Conformal Prediction

---

# Results

Example Test Performance

| Metric | Score |
|---------|--------|
| Accuracy | **86.42%** |
| Weighted Precision | **74.68%** |
| Weighted Recall | **86.42%** |
| Weighted F1 | **80.12%** |
| Ordinal MAE | **0.1481** |
| Conformal Coverage | **92.59%** |

The conformal predictor achieved

- 92.59% empirical coverage

while maintaining

- Average prediction set size of only 1.10 classes,

indicating informative uncertainty estimates.

---

# Key Features

✔ Explainable AI

✔ Rough Set Theory

✔ Adaptive Learning

✔ Fuzzy Inference

✔ Conformal Prediction

✔ Uncertainty Quantification

✔ Rule-based Classification

✔ Physically Interpretable Rules

✔ Robust Cross Validation

✔ Baseline Benchmarking

---

# Project Structure

```
FC-VPRS-ML/
│
├── data/
│   └── hydro_ammina_converted__1_.csv
│
├── rsml_script.py
│
├── README.md
│
└── requirements.txt
```

---

# Installation

```bash
git clone https://github.com/yourusername/FC-VPRS-ML.git

cd FC-VPRS-ML

pip install -r requirements.txt
```

---

# Running

```bash
python rsml_script.py
```

---

# Output

The script automatically generates:

- Classification Metrics
- Confusion Matrix
- Conformal Prediction Coverage
- Average Prediction Set Size
- Human-readable Decision Rules
- Cross-validation Benchmark Results

---

# Applications

- Hydrochar adsorption prediction
- Environmental engineering
- Explainable AI (XAI)
- Scientific decision support
- Materials science
- Sustainable water treatment research

---

# Future Work

- Multi-objective optimization
- Deep Rough Neural Networks
- Multi-label adsorption prediction
- Automatic feature selection
- Larger adsorption datasets
- Web-based visualization dashboard

---

# Citation

If you use this work in your research, please cite the repository.

---

# Author

**Tuli Sen**

B.Tech in Mathematics and Computing

Central University of Karnataka,
Summer Intern at NIT ROURKELA
