#!/usr/bin/env python3
"""
Cuisine prediction with LightGBM (R5), on the split shared with R4 (splits/cuisine_split.csv).

Idea: LightGBM builds decision trees one after the other, each tree correcting the errors of the
previous ones (gradient boosting). "Light" = histogram-based splits and leaf-wise growth, which is
fast on our sparse 0/1 ingredient matrix.

Features: the ingredients only (ingredients_v2, 0/1). Tags, names and descriptions are NOT used: the
label comes from the tags, and names often contain the cuisine ("Italian meatballs").

Usage (repo root)
  python lgbm/lgbm_cuisine.py tune    train -> val: choose the parameters (never reads the test labels)
  python lgbm/lgbm_cuisine.py test    refit on train + val, evaluate ONCE on test, save results and figure
  python lgbm/lgbm_cuisine.py plot    redraw the figure from the saved results

Steps of "tune"
  1. Ingredient matrix, vocabulary from train only
  2. Logistic regression baseline: choose C on val
  3. LightGBM: try 36 parameter settings, number of trees by early stopping on val macro F1
  4. Save results/lgbm_val.csv and results/lgbm_params.json
Steps of "test"
  5. Refit the chosen models on train + val (vocabulary from train + val)
  6. Predict test: LightGBM, logistic regression, majority class
  7. Accuracy, macro F1, F1 per cuisine, confusion matrix, paired bootstrap LightGBM - logistic
  8. Save everything in lgbm/results/
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ------------------------------------------------------------------ settings
HERE        = Path(__file__).resolve().parent
REPO        = HERE.parent
SPLIT_FILE  = REPO / "splits" / "cuisine_split.csv"      # written by R4, never recomputed here
RESULTS_DIR = HERE / "results"
SEED        = 42
MAX_TREES   = 1500       # upper limit, early stopping decides
PATIENCE    = 75         # stop after 75 rounds without a better val macro F1
N_BOOT      = 2000       # bootstrap resamples of the test set
LR_C_VALUES = [0.1, 0.3, 1.0, 3.0]

# Same for every LightGBM run. class_weight="balanced": agreed with R4 (italian is 4.7 x caribbean)
BASE = dict(objective="multiclass", class_weight="balanced", learning_rate=0.05,
            subsample=0.8, subsample_freq=1, random_state=SEED, deterministic=True,
            force_col_wise=True, n_jobs=-1, verbose=-1)
# 4 tree sizes x 3 leaf sizes x 3 feature shares = 36 settings. A first, smaller grid (7-31 leaves,
# 5-20 recipes per leaf, 0.2-0.6 of the features) had its best point on the edge, so it was moved:
# many ingredients are rare, and small leaves are needed to use them.
GRID = [dict(num_leaves=leaves, min_child_samples=mcs, colsample_bytree=cols)
        for leaves in (15, 31, 63, 127) for mcs in (1, 2, 5) for cols in (0.05, 0.1, 0.2)]

start = time.time()


def step(msg):
    print(f"\n[{time.time() - start:5.0f} s] {msg}", flush=True)


# ------------------------------------------------------------------ data
def find_recipes_file():
    """processed/recipes.parquet in a data* folder inside the repo or next to it."""
    for root in (REPO, REPO.parent):
        matches = sorted(p for d in root.glob("data*") for p in d.rglob("processed/recipes.parquet"))
        if matches:
            return matches[0]
    raise FileNotFoundError("processed/recipes.parquet not found: run preprocessing/preprocessing.py first")


def load_data():
    """id, cuisine, split, ingredients_v2 for the 40,912 recipes of the shared split."""
    split = pd.read_csv(SPLIT_FILE)
    recipes = pd.read_parquet(find_recipes_file(), columns=["id", "ingredients_v2"])
    data = split.merge(recipes, on="id", how="left", validate="one_to_one")
    if data["ingredients_v2"].isna().any():
        raise ValueError("some recipes of the split are missing: rerun preprocessing/preprocessing.py")
    return data


def unique_items(ingredients):                   # used as the CountVectorizer analyzer
    return sorted(set(ingredients))


def ingredient_matrix(fit_df, other_df):
    """0/1 recipe x ingredient matrices. Only the ingredients of fit_df become columns."""
    vec = CountVectorizer(analyzer=unique_items, binary=True, dtype=np.float32)
    return vec.fit_transform(fit_df["ingredients_v2"]), vec.transform(other_df["ingredients_v2"]), vec


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}


def logistic(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def val_macro_f1(y_true, proba):
    """Early-stopping metric for LightGBM (y_true = class numbers, proba = one column per class)."""
    return "macro_f1", f1_score(y_true, proba.argmax(axis=1), average="macro"), True


# ------------------------------------------------------------------ tune
def tune():
    step("1. Loading the shared split and building the ingredient matrix")
    data = load_data()
    train, val = data[data["split"] == "train"], data[data["split"] == "val"]
    X_train, X_val, _ = ingredient_matrix(train, val)
    y_train, y_val = train["cuisine"], val["cuisine"]
    print(f"   train {len(train):,}  |  val {len(val):,}  |  {X_train.shape[1]:,} ingredients")
    rows = []

    step("2. Logistic regression: C on val")
    for C in LR_C_VALUES:
        rows.append(dict(model="logistic_regression", C=C, **scores(y_val, logistic(C).fit(X_train, y_train).predict(X_val))))
        print(f"   C={C:<4}  acc {rows[-1]['accuracy']:.4f}  macro F1 {rows[-1]['macro_f1']:.4f}", flush=True)

    step(f"3. LightGBM: {len(GRID)} settings, early stopping on val macro F1")
    for params in GRID:
        t = time.time()
        model = lgb.LGBMClassifier(**BASE, **params, n_estimators=MAX_TREES, metric="None")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=val_macro_f1,
                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        rows.append(dict(model="lightgbm", **params, n_trees=model.best_iteration_, seconds=round(time.time() - t, 1),
                         **scores(y_val, model.predict(X_val))))
        print(f"   {params}  trees {model.best_iteration_}  acc {rows[-1]['accuracy']:.4f}  "
              f"macro F1 {rows[-1]['macro_f1']:.4f}  ({rows[-1]['seconds']:.0f} s)", flush=True)

    step("4. Saving the chosen parameters")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "lgbm_val.csv", index=False, float_format="%.4f")
    best_lr = table.loc[table[table["model"] == "logistic_regression"]["macro_f1"].idxmax()]
    best = table.loc[table[table["model"] == "lightgbm"]["macro_f1"].idxmax()]
    chosen = {"logistic_C": float(best_lr["C"]),
              "lightgbm": {"num_leaves": int(best["num_leaves"]), "min_child_samples": int(best["min_child_samples"]),
                           "colsample_bytree": float(best["colsample_bytree"]), "n_estimators": int(best["n_trees"])}}
    (RESULTS_DIR / "lgbm_params.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print("   chosen on val:", chosen)
    if best["n_trees"] > MAX_TREES - PATIENCE:
        print("   WARNING: early stopping did not trigger, raise MAX_TREES")


# ------------------------------------------------------------------ test
def bootstrap_difference(y_true, pred_a, pred_b, classes):
    """Paired bootstrap over test recipes: 95 % interval of macro F1(a) - macro F1(b)."""
    rng = np.random.default_rng(SEED)
    code = {c: i for i, c in enumerate(classes)}
    t, a, b = (np.array([code[c] for c in x]) for x in (y_true, pred_a, pred_b))
    k = len(classes)

    def macro_f1(true, pred):
        cm = np.bincount(true * k + pred, minlength=k * k).reshape(k, k)
        return np.mean(2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1))

    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(t), len(t))
        diffs.append(macro_f1(t[i], a[i]) - macro_f1(t[i], b[i]))
    return macro_f1(t, a) - macro_f1(t, b), *np.quantile(diffs, [0.025, 0.975])


def test():
    chosen = json.loads((RESULTS_DIR / "lgbm_params.json").read_text())
    step("5. Refit on train + val")
    data = load_data()
    fit, held_out = data[data["split"] != "test"], data[data["split"] == "test"]
    X_fit, X_test, vec = ingredient_matrix(fit, held_out)
    classes = sorted(data["cuisine"].unique())            # alphabetical order, as in splits/README.md
    print(f"   train + val {len(fit):,}  |  test {len(held_out):,}  |  {X_fit.shape[1]:,} ingredients")
    model = lgb.LGBMClassifier(**BASE, **chosen["lightgbm"]).fit(X_fit, fit["cuisine"])
    lr = logistic(chosen["logistic_C"]).fit(X_fit, fit["cuisine"])

    step("6. Predict test")
    majority = fit["cuisine"].value_counts().idxmax()
    preds = {"lightgbm": model.predict(X_test), "logistic_regression": lr.predict(X_test),
             "majority_class": np.full(len(held_out), majority)}

    step("7. Evaluate")
    y_test = held_out["cuisine"].to_numpy()
    results = pd.DataFrame([dict(model=m, n_test=len(y_test), **scores(y_test, p)) for m, p in preds.items()])
    diff, low, high = bootstrap_difference(y_test, preds["lightgbm"], preds["logistic_regression"], classes)
    per_class = pd.DataFrame({f"f1_{m}": f1_score(y_test, p, labels=classes, average=None, zero_division=0)
                              for m, p in preds.items()}, index=pd.Index(classes, name="cuisine"))
    per_class.insert(0, "n_test", [int((y_test == c).sum()) for c in classes])
    cm = pd.DataFrame(confusion_matrix(y_test, preds["lightgbm"], labels=classes),
                      index=["true_" + c for c in classes], columns=["pred_" + c for c in classes])
    gain = pd.DataFrame({"ingredient": vec.get_feature_names_out(),
                         "gain": model.booster_.feature_importance("gain"),
                         "n_splits": model.booster_.feature_importance("split")})
    gain["gain_share"] = gain["gain"] / gain["gain"].sum()

    step("8. Saving results")
    results.to_csv(RESULTS_DIR / "lgbm_test.csv", index=False, float_format="%.4f")
    pd.DataFrame([dict(comparison="lightgbm - logistic_regression", metric="macro_f1", difference=diff,
                       ci95_low=low, ci95_high=high, n_bootstrap=N_BOOT)]).to_csv(
        RESULTS_DIR / "lgbm_bootstrap.csv", index=False, float_format="%.4f")
    per_class.to_csv(RESULTS_DIR / "lgbm_per_class.csv", float_format="%.4f")
    cm.to_csv(RESULTS_DIR / "lgbm_confusion_matrix.csv")
    gain.sort_values("gain", ascending=False).head(30).to_csv(
        RESULTS_DIR / "lgbm_top_ingredients.csv", index=False, float_format="%.4f")
    pd.DataFrame({"id": held_out["id"], "cuisine": y_test, "pred_lightgbm": preds["lightgbm"],
                  "pred_logistic_regression": preds["logistic_regression"]}).to_csv(
        RESULTS_DIR / "lgbm_test_predictions.csv", index=False)      # lets R4 compare recipe by recipe
    print(results.round(4).to_string(index=False))
    print(f"\nLightGBM - logistic regression, macro F1: {diff:+.4f}  (95 % interval {low:+.4f} to {high:+.4f})")
    print("\n" + per_class.round(3).to_string())
    print("\ntop ingredients by gain:", ", ".join(gain.nlargest(15, "gain")["ingredient"]))
    plot()


# ------------------------------------------------------------------ figure
def plot():
    """(a) confusion matrix of LightGBM (row %), (b) F1 per cuisine, LightGBM vs logistic regression."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = pd.read_csv(RESULTS_DIR / "lgbm_confusion_matrix.csv", index_col=0)
    per_class = pd.read_csv(RESULTS_DIR / "lgbm_per_class.csv", index_col=0)
    classes = per_class.index.tolist()
    short = [c.replace("southern-united-states", "southern-us") for c in classes]
    share = cm.to_numpy() / cm.to_numpy().sum(axis=1, keepdims=True)
    n = len(classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax1.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(n):
            if share[i, j] >= 0.005:
                ax1.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=6,
                         color="white" if share[i, j] > 0.55 else "black")
    ax1.set_xticks(range(n), short, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.5)
    ax1.set_yticks(range(n), short, fontsize=6.5)
    ax1.set_xlabel("predicted cuisine", fontsize=7)
    ax1.set_ylabel("true cuisine", fontsize=7)
    ax1.tick_params(length=0)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.set_title("(a) LightGBM confusion matrix, % of each row", fontsize=7.5, loc="left")

    order = per_class.sort_values("f1_lightgbm").index
    y = np.arange(n)
    a, b = per_class.loc[order, "f1_lightgbm"], per_class.loc[order, "f1_logistic_regression"]
    ax2.hlines(y, np.minimum(a, b), np.maximum(a, b), color="#c3c2b7", lw=1.4, zorder=1)
    ax2.scatter(b, y, s=26, color="#eb6834", marker="s", edgecolor="white", lw=0.8, label="logistic regression", zorder=2)
    ax2.scatter(a, y, s=32, color="#2a78d6", marker="o", edgecolor="white", lw=0.8, label="LightGBM", zorder=3)
    ax2.set_yticks(y, [c.replace("southern-united-states", "southern-us") for c in order], fontsize=6.5)
    ax2.set_xlim(0.2, 1)
    ax2.set_xlabel("test F1", fontsize=7)
    ax2.tick_params(axis="x", labelsize=6.5)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color="#e6e5e1", lw=0.6)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], labels[::-1], fontsize=6.5, frameon=False, loc="upper left", handletextpad=0.2)
    ax2.set_title("(b) F1 per cuisine", fontsize=7.5, loc="left")

    fig.tight_layout(w_pad=1.2)
    fig.savefig(RESULTS_DIR / "lgbm_figure.pdf")                # for Overleaf
    fig.savefig(RESULTS_DIR / "lgbm_figure.png", dpi=250)       # for quick viewing
    plt.close(fig)
    print(f"   saved {RESULTS_DIR / 'lgbm_figure.pdf'} and .png")


if __name__ == "__main__":
    actions = {"tune": tune, "test": test, "plot": plot}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        sys.exit(f"usage: python {Path(__file__).name} {'|'.join(actions)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    actions[sys.argv[1]]()
