"""Model scoring, explanation and cost logic. No Streamlit imports, so it can be unit-tested."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
BINARY = ["HighBP", "HighChol", "HeartDiseaseorAttack", "DiffWalk"]
# BRFSS 5-year age bands: code -> lower bound in years (1 = 18-24 ... 13 = 80+)
AGE_LOWER = {1: 18, 2: 25, 3: 30, 4: 35, 5: 40, 6: 45, 7: 50, 8: 55, 9: 60, 10: 65, 11: 70, 12: 75, 13: 80}


def load_model():
    return json.loads((ROOT / "model" / "model.json").read_text())


def load_metrics():
    return json.loads((ROOT / "model" / "metrics.json").read_text())


def load_scores():
    return pd.read_csv(ROOT / "data" / "test_scores.csv.gz")


def age_to_code(age_years: float) -> int:
    code = 1
    for c, lo in AGE_LOWER.items():
        if age_years >= lo:
            code = c
    return code


def age_code_label(code: int) -> str:
    lo = AGE_LOWER[int(code)]
    return f"{lo}+" if code == 13 else f"{lo}-{AGE_LOWER[int(code) + 1] - 1}"


def bmi_from(height_cm: float, weight_kg: float) -> float:
    return weight_kg / ((height_cm / 100.0) ** 2)


def _logit(X: pd.DataFrame, m: dict) -> np.ndarray:
    x = X[m["features"]].to_numpy(dtype=float)
    z = (x - np.array(m["mean"])) / np.array(m["scale"])
    return z @ np.array(m["coef"]) + m["intercept"]


def predict_proba(X: pd.DataFrame, m: dict) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(X, m)))


def contributions(case: dict, m: dict) -> dict:
    """Per-feature push on the log-odds relative to an average respondent (exact for logistic regression)."""
    return {f: float(c * (case[f] - mu) / s)
            for f, c, mu, s in zip(m["features"], m["coef"], m["mean"], m["scale"])}


def threshold_from_costs(c_fp: float, c_fn: float) -> float:
    """Bayes-optimal threshold for calibrated probabilities: flag when p * C_FN > (1 - p) * C_FP."""
    return c_fp / (c_fp + c_fn)


def tier(p: float, t: float) -> str:
    if p >= t:
        return "high"
    return "mid" if p >= 0.5 * t else "low"


def confusion(scores: pd.DataFrame, t: float) -> dict:
    pred = scores["p"].to_numpy() >= t
    y = scores["y"].to_numpy() == 1
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    n = max(len(y), 1)
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "flagged": (tp + fp) / n, "fp_rate": fp / n, "fn_rate": fn / n}


def cost_curve(scores: pd.DataFrame, c_fp: float, c_fn: float, step: float = 0.005) -> pd.DataFrame:
    p = scores["p"].to_numpy()
    y = scores["y"].to_numpy() == 1
    n = len(y)
    rows = []
    for t in np.arange(0.02, 0.80, step):
        pred = p >= t
        rows.append((t, 1000 * (c_fp * (pred & ~y).sum() + c_fn * (~pred & y).sum()) / n))
    return pd.DataFrame(rows, columns=["threshold", "cost"])


def expected_cost_per_1000(scores: pd.DataFrame, t: float, c_fp: float, c_fn: float) -> float:
    c = confusion(scores, t)
    return 1000 * (c_fp * c["fp_rate"] + c_fn * c["fn_rate"])


def normalize_batch(df: pd.DataFrame, features: list):
    """Accept BRFSS-coded columns, or friendlier AgeYears / Height_cm + Weight_kg. Returns (clean, valid_mask, missing)."""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    lower = {c.lower(): c for c in d.columns}
    work = pd.DataFrame(index=d.index)
    for f in features:
        if f.lower() in lower:
            work[f] = pd.to_numeric(d[lower[f.lower()]], errors="coerce")
    if "Age" not in work and "ageyears" in lower:
        work["Age"] = pd.to_numeric(d[lower["ageyears"]], errors="coerce").apply(
            lambda a: age_to_code(a) if pd.notna(a) and a >= 18 else np.nan)
    if "BMI" not in work and "height_cm" in lower and "weight_kg" in lower:
        work["BMI"] = (pd.to_numeric(d[lower["weight_kg"]], errors="coerce")
                       / (pd.to_numeric(d[lower["height_cm"]], errors="coerce") / 100.0) ** 2)
    missing = [f for f in features if f not in work]
    if missing:
        return work, None, missing
    ok = work[features].notna().all(axis=1)
    for f in BINARY:
        ok &= work[f].isin([0, 1])
    ok &= work["GenHlth"].between(1, 5) & work["Age"].between(1, 13) & work["BMI"].between(12, 98)
    return work, ok, []
