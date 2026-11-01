"""Reality check of Perez Castillo et al. (2024) on TAWOS.

Three experiments, all fixed before running (seed 42):

E1  The paper's protocol. End-of-sprint R1-R11, the same 7 classifiers,
    stratified 10-fold cross-validation with sprints shuffled at random
    (the paper does not give the number of folds). Metrics as in the paper
    (weighted recall, F1, accuracy, precision), plus macro F1 and the recall
    of the Poor class, which the paper calls the key class. Baselines: always
    the majority class, and logistic regression on R1 alone.
E2  Same features, honest splits: a time split (each project's first 70% of
    sprints train, the last 30% test) and leave-one-project-out.
E3  Early warning, which is what the paper says the tool is for ("detect
    sprints that may require immediate intervention"). Features use only the
    days up to 25% or 50% of the sprint; the label is still the end-of-sprint
    TAG. Time split.

Everything is run twice: with our main TAG and with the "pace only" TAG
(see features.py).

    python src/reality_check.py      (needs data/burndown_daily.csv)
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CLASSES, R_COLS, add_tag, build_table  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = 1000


def models() -> dict:
    """Library defaults: the paper gives no settings. Scaling for the distance/gradient models."""
    sc = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    return {
        "KNN": sc(KNeighborsClassifier()),
        "SVM": sc(SVC(random_state=SEED)),
        "MLP": sc(MLPClassifier(max_iter=2000, random_state=SEED)),
        "RF": RandomForestClassifier(random_state=SEED),
        "NB": GaussianNB(),
        "LR": sc(LogisticRegression(max_iter=2000)),
        "AB": AdaBoostClassifier(random_state=SEED),
    }


def baselines() -> dict:
    return {
        "Majority class": (DummyClassifier(strategy="most_frequent"), R_COLS),
        "LR on R1 only": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), ["R1"]),
    }


def scores(y, p) -> dict:
    return {
        "recall_w": recall_score(y, p, average="weighted", zero_division=0),
        "f1_w": f1_score(y, p, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y, p),
        "precision_w": precision_score(y, p, average="weighted", zero_division=0),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "poor_recall": recall_score(y == 0, p == 0, zero_division=0),
        "poor_precision": precision_score(y == 0, p == 0, zero_division=0),
    }


def boot_ci(y, p, metric, rng) -> tuple[float, float]:
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        vals.append(metric(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]).round(3))


def run_set(name, table, fit_pred, cols_all=R_COLS):
    """fit_pred(model, cols) -> (y_true, y_pred). Returns rows for every model and baseline."""
    rows = []
    entries = [(m, est, cols_all, "model") for m, est in models().items()]
    entries += [(m, est, cols, "baseline") for m, (est, cols) in baselines().items()]
    for m, est, cols, kind in entries:
        y, p = fit_pred(est, cols)
        rows.append({"experiment": name, "model": m, "kind": kind, "n": len(y), **scores(y, p)})
    return rows


def run_variant(label_name, column, tables):
    """All experiments for one label definition. tables: {cut-off fraction: feature table}."""
    end_table, cuts = add_tag(tables[1.0], column)
    tags = end_table.set_index("sprint_id")["TAG"]
    out, ci_rows = [], []
    rng = np.random.default_rng(SEED)

    # E1: paper protocol, random stratified 10-fold CV over all sprints
    X, y = end_table, end_table["TAG"].to_numpy()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    out += run_set("E1 random 10-fold CV (paper protocol)", end_table,
                   lambda est, cols: (y, cross_val_predict(clone(est), X[cols], y, cv=cv)))

    # E2a and E3: time split at several cut-offs
    def time_split_fit(table):
        tr, te = table[table["split"] == "train"], table[table["split"] == "test"]

        def f(est, cols):
            m = clone(est).fit(tr[cols], tr["TAG"])
            return te["TAG"].to_numpy(), m.predict(te[cols])
        return f

    for frac, name in [(1.0, "E2 time split, end of sprint"),
                       (0.5, "E3 time split, features at 50% of sprint"),
                       (0.25, "E3 time split, features at 25% of sprint")]:
        t = end_table if frac == 1.0 else tables[frac].assign(TAG=lambda d: d["sprint_id"].map(tags))
        rows = run_set(name, t, time_split_fit(t))
        out += rows
        f = time_split_fit(t)
        best = max((r for r in rows if r["kind"] == "model"), key=lambda r: r["f1_macro"])["model"]
        for m, (est, cols) in {best: (models()[best], R_COLS), **baselines()}.items():
            yt, yp = f(est, cols)
            ci_rows.append({"label": label_name, "experiment": name, "model": m,
                            "accuracy": round(accuracy_score(yt, yp), 3),
                            "accuracy_ci": boot_ci(yt, yp, accuracy_score, rng),
                            "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 3),
                            "f1_macro_ci": boot_ci(yt, yp, lambda a, b: f1_score(a, b, average="macro",
                                                                                  zero_division=0), rng),
                            "poor_recall": round(recall_score(yt == 0, yp == 0, zero_division=0), 3)})

    # E2b: leave one project out
    projects = sorted(end_table["project"].unique())

    def lopo(est, cols):
        ys, ps = [], []
        for pr in projects:
            tr, te = end_table[end_table["project"] != pr], end_table[end_table["project"] == pr]
            m = clone(est).fit(tr[cols], tr["TAG"])
            ys.append(te["TAG"].to_numpy())
            ps.append(m.predict(te[cols]))
        return np.concatenate(ys), np.concatenate(ps)
    out += run_set("E2 leave one project out, end of sprint", end_table, lopo)

    for r in out:
        r["label"] = label_name
    shares = end_table["TAG"].value_counts(normalize=True).sort_index()
    info = {"tag_cut_points": [round(float(c), 4) for c in cuts],
            "tag_shares": {CLASSES[i]: round(float(v), 3) for i, v in shares.items()}}
    return out, ci_rows, end_table, info


def main():
    RES.mkdir(exist_ok=True)
    daily = pd.read_csv(ROOT / "data" / "burndown_daily.csv", parse_dates=["start_date"])
    tables = {f: build_table(daily, f) for f in (1.0, 0.5, 0.25)}

    out, ci_rows, labels = [], [], {}
    for label_name, column in [("main: pace + completion", "score"), ("sensitivity: pace only", "pace")]:
        o, c, end_table, info = run_variant(label_name, column, tables)
        out += o
        ci_rows += c
        labels[label_name] = info
        if column == "score":
            end_table.to_csv(RES / "sprint_table_end.csv", index=False)

    res = pd.DataFrame(out)
    res = res[["label"] + [c for c in res.columns if c != "label"]]
    num = res.select_dtypes("number").columns.drop("n")
    res[num] = res[num].round(3)
    res.to_csv(RES / "reality_check_results.csv", index=False)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(RES / "reality_check_intervals.csv", index=False)

    t = tables[1.0]
    summary = {
        "sprints": int(len(t)), "train": int((t["split"] == "train").sum()),
        "test": int((t["split"] == "test").sum()), "projects": sorted(t["project"].unique()),
        "median_sprint_days": round(float(t["n_days"].median()), 2),
        "labels": labels,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "seed": SEED,
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=2))
    print(res.drop(columns=["kind"]).to_string(index=False))
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
