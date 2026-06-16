"""Train and evaluate models that predict a vessel's operational CO2 efficiency
grade (A-E) from its structural and technical attributes (EU MRV data).

Run:  python src/train.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import build_features, NUMERIC, CATEGORICAL

DATA = Path(__file__).resolve().parents[1] / "data" / "vessels_mrv.csv"


def make_preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def candidate_models():
    return {
        "baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": LogisticRegression(max_iter=1000),
        "random forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "gradient boosting": GradientBoostingClassifier(random_state=42),
    }


def main():
    df = pd.read_csv(DATA)
    X, y = build_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    pre = make_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Samples: {len(X)} | classes: {sorted(y.unique())}\n")
    print("5-fold CV accuracy (train):")
    best_name, best_score, best_pipe = None, -1, None
    for name, model in candidate_models().items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
        print(f"  {name:26s} {scores.mean():.3f} +/- {scores.std():.3f}")
        if name != "baseline (most frequent)" and scores.mean() > best_score:
            best_name, best_score, best_pipe = name, scores.mean(), pipe

    print(f"\nBest model: {best_name}")
    best_pipe.fit(X_tr, y_tr)
    y_pred = best_pipe.predict(X_te)
    print("\nHeld-out test report:")
    print(classification_report(y_te, y_pred))
    print("Confusion matrix (rows=true A-E):")
    print(confusion_matrix(y_te, y_pred, labels=sorted(y.unique())))

    # permutation importance on the held-out set
    r = permutation_importance(best_pipe, X_te, y_te, n_repeats=10,
                               random_state=42, scoring="accuracy")
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    print("\nPermutation importance (accuracy drop):")
    print(imp.round(4).to_string())
    return best_pipe


if __name__ == "__main__":
    main()
