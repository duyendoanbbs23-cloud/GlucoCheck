"""Run with: python -m pytest -q   (or python tests/test_core.py)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core  # noqa: E402

M = core.load_model()
SC = core.load_scores()


def row(**kw):
    base = dict(GenHlth=2, Age=6, BMI=25, HighBP=0, HighChol=0, HeartDiseaseorAttack=0, DiffWalk=0)
    return pd.DataFrame([{**base, **kw}])


def test_threshold_rule():
    assert abs(core.threshold_from_costs(1, 5) - 1 / 6) < 1e-12
    assert core.threshold_from_costs(1, 1) == 0.5


def test_risk_goes_up_with_risk_factors():
    low, high = core.predict_proba(row(), M)[0], core.predict_proba(row(GenHlth=5, HighBP=1, BMI=38, Age=11), M)[0]
    assert 0 < low < high < 1


def test_contributions_sum_to_logit_shift():
    case = row(GenHlth=4, BMI=33, HighBP=1).iloc[0].to_dict()
    total = sum(core.contributions(case, M).values())
    base = core._logit(pd.DataFrame([M["population_mean"]]), M)[0]
    assert abs(core._logit(pd.DataFrame([case]), M)[0] - base - total) < 1e-9


def test_higher_threshold_flags_fewer():
    assert core.confusion(SC, 0.3)["flagged"] < core.confusion(SC, 0.1)["flagged"]


def test_batch_validation():
    df = pd.DataFrame({"GenHlth": [3, 9], "Age": [5, 5], "BMI": [27, 27], "HighBP": [0, 0],
                       "HighChol": [0, 0], "HeartDiseaseorAttack": [0, 0], "DiffWalk": [0, 1]})
    _, ok, missing = core.normalize_batch(df, M["features"])
    assert not missing and ok.tolist() == [True, False]
    _, ok, missing = core.normalize_batch(df.drop(columns="BMI"), M["features"])
    assert missing == ["BMI"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
