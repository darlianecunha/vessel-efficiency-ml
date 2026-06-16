import sys, math
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import parse_technical_efficiency, build_features, LEAKAGE, TARGET


def test_parse_te_numeric():
    assert parse_technical_efficiency("EEXI (3.24 gCO2/t·nm)") == ("EEXI", 3.24)


def test_parse_te_not_applicable():
    metric, val = parse_technical_efficiency("Not Applicable")
    assert metric == "NotApplicable" and math.isnan(val)


def test_build_features_no_leakage():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "vessels_mrv.csv")
    X, y = build_features(df)
    # the CO2 fields that define the grade must NOT be among the features
    for col in LEAKAGE + [TARGET]:
        assert col not in X.columns
    assert len(X) == len(y) > 0
    assert set(y.unique()) <= {"A", "B", "C", "D", "E"}
