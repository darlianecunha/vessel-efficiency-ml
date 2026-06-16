"""Feature engineering for the vessel efficiency model.

Target:   efficiency_grade (A-E), the operational CO2 efficiency band.
Features: vessel STRUCTURAL + technical attributes only. The CO2 fields that
define the grade (co2_per_distance_kg_nm, co2_total_t) are excluded to avoid
target leakage.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

TARGET = "efficiency_grade"
LEAKAGE = ["co2_per_distance_kg_nm", "co2_total_t"]
DROP = ["imo", "name", "in_mrv"]

_NUM = re.compile(r"([0-9]+\.?[0-9]*)")


def parse_technical_efficiency(value: str):
    """Return (metric_type, numeric_value) from strings like
    'EEXI (3.24 gCO2/t·nm)' or 'Not Applicable'."""
    if not isinstance(value, str) or "not applicable" in value.lower():
        return "NotApplicable", np.nan
    metric = value.split("(")[0].strip() or "Other"
    m = _NUM.search(value)
    return metric, (float(m.group(1)) if m else np.nan)


def group_flags(s: pd.Series, top: int = 10) -> pd.Series:
    keep = s.value_counts().nlargest(top).index
    return s.where(s.isin(keep), "Other")


def build_features(df: pd.DataFrame):
    """Return (X, y) ready for the modelling pipeline."""
    df = df.copy()
    te = df["technical_efficiency"].apply(parse_technical_efficiency)
    df["te_metric"] = te.apply(lambda t: t[0])
    df["te_value"] = te.apply(lambda t: t[1])
    df["flag_grouped"] = group_flags(df["flag_registry"])

    y = df[TARGET].astype(str)
    feature_cols = ["ship_type", "dwt", "dwt_size_class", "flag_grouped",
                    "te_metric", "te_value"]
    X = df[feature_cols]
    return X, y


NUMERIC = ["dwt", "te_value"]
CATEGORICAL = ["ship_type", "dwt_size_class", "flag_grouped", "te_metric"]
