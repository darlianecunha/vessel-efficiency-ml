# vessel-efficiency-ml

> Supervised machine-learning pipeline that predicts a ship's operational CO2 efficiency grade (A-E) from its structural and technical attributes, using EU MRV emissions data.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org)

## Objective

Can we predict a vessel's **operational efficiency grade (A-E)** from what we know about the ship, *without* using the CO2 figures that define the grade? This is a genuine, non-trivial classification problem and a useful screening tool for fleet decarbonisation.

The CO2 fields that define the grade (`co2_per_distance_kg_nm`, `co2_total_t`) are **deliberately excluded** to avoid target leakage.

## Data

EU MRV (Monitoring, Reporting and Verification) vessel records. The bundled `data/vessels_mrv.csv` contains 1,570 vessels with complete attributes; the model scales to the full THETIS-MRV dataset (tens of thousands of ships) without code changes.

Required columns: `ship_type`, `dwt`, `dwt_size_class`, `efficiency_grade`, `flag_registry`, `technical_efficiency` (e.g. "EEXI (3.24 gCO2/t·nm)"), plus the excluded CO2 fields.

To use a larger dataset, simply replace `data/vessels_mrv.csv` (same columns) and re-run.

## Method

- Feature engineering: parse the technical-efficiency index (EEXI/EIV value + type), group rare flags, drop identifiers and leakage.
- Pipeline: median/mode imputation, scaling, one-hot encoding (`handle_unknown='ignore'`).
- Models compared: baseline, logistic regression, random forest, gradient boosting.
- Evaluation: stratified 5-fold cross-validation, held-out test report, confusion matrix, permutation importance, and a learning curve.

## Results (bundled 1,570-vessel sample)

| Model | CV accuracy |
|---|---|
| Baseline (most frequent) | 0.21 |
| Logistic regression | 0.40 |
| Random forest | 0.42 |
| Gradient boosting | **0.42** |

The best model roughly **doubles the baseline** on 5 balanced classes. The confusion matrix shows it captures the A→E ordering (extremes predicted best). DWT is the most predictive feature, followed by the technical-efficiency index. Predicting the grade without the operational CO2 is intentionally hard, so this is an honest, leakage-free result.

## Run it

```bash
pip install -r requirements.txt
python src/train.py                 # full pipeline + metrics
pytest                              # tests
jupyter notebook notebooks/vessel_efficiency_ml.ipynb
```

## Repository structure

```
vessel-efficiency-ml/
├── src/
│   ├── features.py     # feature engineering (leakage-free)
│   └── train.py        # pipeline, CV, evaluation, importance
├── notebooks/          # narrative analysis + learning curve
├── data/               # EU MRV sample (swap for the full dataset)
└── tests/              # unit tests
```

## License

Code: MIT · Data: EU MRV public data under its own terms.
