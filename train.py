"""Train the screening model and export everything the app needs.

Usage:
    python train.py path/to/diabetes_binary_health_indicators_BRFSS2015.csv [--out DIR]

Writes into DIR (default: this folder):
    model/model.json         logistic-regression parameters (plain JSON, no pickle)
    model/metrics.json       overall, per-group (fairness), calibration, selection curve
    data/test_scores.csv.gz  held-out predictions + group labels (drives the live simulator)
    data/sample_batch.csv    200 anonymous rows for trying the batch page
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "Diabetes_binary"
DEFAULT_THRESHOLD = 0.15

# Inputs a person knows at home. Income/Education are audit-only (never model inputs);
# care-access columns say little about the body and could encode inequality.
EXCLUDED = {"Income", "Education", "CholCheck", "AnyHealthcare", "NoDocbcCost"}

# Final 7 questions. Chosen from the forward-selection curve (AUC plateaus after ~5),
# with three deliberate choices:
#   * HvyAlcoholConsump scores well but looks *protective* in this data -> would mislead users.
#   * Sex is a sensitive attribute and adds ~0.0006 AUC -> left out; used only to audit fairness.
#   * HeartDiseaseorAttack and DiffWalk are kept as cheap, familiar questions that matter clinically.
FEATURES = ["GenHlth", "Age", "BMI", "HighBP", "HighChol", "HeartDiseaseorAttack", "DiffWalk"]


def pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def bucket_income(i):
    return "low" if i <= 3 else "mid" if i <= 6 else "high"           # <$20k / $20-50k / >=$50k


def bucket_edu(e):
    return "no_college" if e <= 4 else "some_college" if e == 5 else "college_grad"


def bucket_age(a):
    return "18-44" if a <= 5 else "45-59" if a <= 8 else "60+"        # BRFSS 5-year bands


def op_metrics(y, p, t):
    pred = p >= t
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, tn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    return {"sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "ppv": float(tp / max(tp + fp, 1)), "flagged": float(pred.mean())}


def main(csv, out):
    df = pd.read_csv(csv)
    y = df.pop(TARGET).astype(int)
    cands = [c for c in df.columns if c not in EXCLUDED]

    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=1, stratify=y_tr)

    # Forward selection on the validation split (test set untouched until reporting)
    sel, curve = [], []
    while len(sel) < len(cands):
        def val_auc(c):
            cols = sel + [c]
            return roc_auc_score(y_val, pipe().fit(X_fit[cols], y_fit).predict_proba(X_val[cols])[:, 1])
        scored = {c: val_auc(c) for c in cands if c not in sel}
        best = max(scored, key=scored.get)
        sel.append(best)
        curve.append({"n": len(sel), "added": best, "auc": round(float(scored[best]), 4)})
        print(curve[-1])

    model = pipe().fit(X_tr[FEATURES], y_tr)
    sc, lr = model[0], model[1]
    p = model.predict_proba(X_te[FEATURES])[:, 1]
    yt = y_te.to_numpy()

    full_lr = pipe().fit(X_tr[cands], y_tr)
    hgb = HistGradientBoostingClassifier(random_state=0).fit(X_tr[cands], y_tr)

    overall = {
        "auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
        "brier": brier_score_loss(yt, p), "prevalence": float(yt.mean()),  # test-set prevalence
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        f"auc_full_hgb_{len(cands)}_features": roc_auc_score(yt, hgb.predict_proba(X_te[cands])[:, 1]),
        f"auc_all_{len(cands)}_lr": roc_auc_score(yt, full_lr.predict_proba(X_te[cands])[:, 1]),
        "default_threshold": DEFAULT_THRESHOLD, **op_metrics(yt, p, DEFAULT_THRESHOLD),
    }

    # Group labels come from columns that are NOT model inputs (audit only)
    scores = pd.DataFrame({
        "p": p.round(4), "y": yt,
        "Income": X_te["Income"].map(bucket_income).values,
        "Education": X_te["Education"].map(bucket_edu).values,
        "Sex": X_te["Sex"].map({0: "female", 1: "male"}).values,
        "Age": X_te["Age"].map(bucket_age).values,
    })

    order = {"Income": ["low", "mid", "high"],
             "Education": ["no_college", "some_college", "college_grad"],
             "Sex": ["male", "female"], "Age": ["18-44", "45-59", "60+"]}
    fairness = []
    for attr, groups in order.items():
        for g in groups:
            sub = scores[scores[attr] == g]
            if sub.y.nunique() < 2:
                continue
            fairness.append({"attribute": attr, "group": g, "n": int(len(sub)),
                             "prevalence": float(sub.y.mean()), "auc": roc_auc_score(sub.y, sub.p),
                             "mean_pred": float(sub.p.mean()),
                             **op_metrics(sub.y.to_numpy(), sub.p.to_numpy(), DEFAULT_THRESHOLD)})

    q = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": yt, "q": q}).groupby("q", observed=True).agg(
        pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).to_dict("records")

    out = Path(out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    model_json = {"features": FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                  "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                  "population_mean": dict(zip(FEATURES, sc.mean_.tolist()))}
    (out / "model" / "model.json").write_text(json.dumps(model_json))
    (out / "model" / "metrics.json").write_text(json.dumps(
        {"overall": overall, "fairness": fairness, "calibration": cal, "selection_curve": curve[:10]},
        default=float))
    scores.to_csv(out / "data" / "test_scores.csv.gz", index=False)
    X_te[FEATURES].sample(200, random_state=0).to_csv(out / "data" / "sample_batch.csv", index=False)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in overall.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=str(Path(__file__).parent))
    a = ap.parse_args()
    main(a.csv, a.out)
