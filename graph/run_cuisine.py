"""Can the communities of the ingredient graph predict a recipe's cuisine? (R4)

Uses the shared split splits/cuisine_split.csv (11 cuisines, train / val / test, same as R5's model).
  1. Ingredient graph from all recipes except the test ones (no label is used to build it), weighted by
     raw count or by PMI, same threshold rule as run_graph.py.
  2. Louvain at several resolutions (more resolution = more, smaller communities).
  3. Each recipe -> share of its ingredients in each community.
  4. Predictors: community -> cuisine table (with / without dividing by the cuisine's share) and a
     logistic regression on the profile. References: majority class, logistic regression on all
     ingredients (one-hot).
  5. Validation picks the predictor. Count and PMI are then compared on test at the SAME number of
     communities (10, 30, 100; the Louvain resolution is searched for each graph), refit on train + val,
     Louvain repeated with 5 seeds. Comparing at equal size matters: at high resolution the count graph
     breaks into almost one community per ingredient, which is no clustering any more.

Usage (repo root): python graph/run_cuisine.py
Outputs: graph/results/cuisine_val.csv, cuisine_val_references.csv, cuisine_test.csv, cuisine_per_class.csv, cuisine_confusion.csv,
cuisine_communities.csv and graph/figures/cuisine_resolution.png
"""
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from sklearn.preprocessing import MultiLabelBinarizer  # noqa: E402

import cuisine  # noqa: E402
import graphs  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
SPLIT = HERE.parent / "splits" / "cuisine_split.csv"
WEIGHTINGS = ["count", "pmi"]
RESOLUTIONS = [0.5, 1, 2, 4, 8, 16, 32]
TARGET_K = [10, 30, 100]
SEEDS = [42, 43, 44, 45, 46]
METHODS = ["table", "table_prior", "logistic"]

# Reference palette (light mode), same colours per weighting as run_graph.py
COLOUR = {"count": "#2a78d6", "pmi": "#1baf7a"}
STYLE = {"table": ":", "table_prior": "--", "logistic": "-"}
INK2, MUTED, SURFACE, GRID, AXIS = "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"


def logistic():
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def fit_predict(method, X_fit, y_fit, X_eval, classes):
    if method == "logistic":
        return logistic().fit(X_fit, y_fit).predict(X_eval)
    table = cuisine.community_cuisine_table(X_fit, y_fit, classes)
    prior = cuisine.class_prior(y_fit, classes)
    return cuisine.predict_from_table(X_eval, table, classes, prior=prior if method == "table_prior" else None,
                                      fallback=classes[int(np.argmax(prior))])


def profiles(partition, parts):
    k = max(partition.values()) + 1
    return {name: cuisine.community_profiles(df["ingredients_v2"], partition, k) for name, df in parts.items()}


def resolution_for_k(G, k_target, seed=graphs.SEED, lo=0.001, hi=500.0, grid=40, steps=20):
    """Louvain resolution whose partition has the number of communities closest to k_target.
    The number of communities grows with the resolution but jumps and is not perfectly monotonic, so:
    a log-spaced grid first, then bisection between the two grid points that bracket k_target."""
    def n_comm(r):
        return max(graphs.louvain(G, seed=seed, resolution=r).values()) + 1

    tried = [(r, n_comm(r)) for r in np.geomspace(lo, hi, grid)]
    below = [t for t in tried if t[1] <= k_target]
    above = [t for t in tried if t[1] >= k_target]
    if below and above:
        a, b = max(below, key=lambda t: t[0])[0], min((t for t in above if t[0] > max(below, key=lambda t: t[0])[0]),
                                                     key=lambda t: t[0], default=above[0])[0]
        for _ in range(steps):
            mid = float(np.sqrt(a * b))
            k = n_comm(mid)
            tried.append((mid, k))
            if k == k_target:
                break
            a, b = (mid, b) if k < k_target else (a, mid)
    return min(tried, key=lambda t: (abs(t[1] - k_target), t[0]))


def onehot_reference(fit, evaluate_on, classes):
    mlb = MultiLabelBinarizer(sparse_output=True).fit(fit["ingredients_v2"])
    known = set(mlb.classes_)
    X_fit = mlb.transform(fit["ingredients_v2"])
    X_eval = mlb.transform(evaluate_on["ingredients_v2"].map(lambda xs: [x for x in xs if x in known]))
    return logistic().fit(X_fit, fit["cuisine"]).predict(X_eval)


def plot_resolution(val: pd.DataFrame, refs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for w in WEIGHTINGS:
        for m in METHODS:
            d = val[(val["weighting"] == w) & (val["method"] == m)].sort_values("n_communities")
            ax.plot(d["n_communities"], d["macro_f1"], color=COLOUR[w], linestyle=STYLE[m], linewidth=2,
                    marker="o", markersize=4, label=f"{w} graph, {m.replace('_', ' + ')}")
    ax.set_xscale("log")
    for name, value in refs.items():
        ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0.01, value, f"  {name} ({value:.2f})", transform=ax.get_yaxis_transform(), fontsize=7,
                color=INK2, va="bottom")
    ax.set_xlabel("number of Louvain communities (log scale)", fontsize=8, color=INK2)
    ax.set_ylabel("macro-F1 on the validation recipes", fontsize=8, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    recipes = graphs.load_recipes()
    split = pd.read_csv(SPLIT)
    labelled = split.merge(recipes[["id", "ingredients_v2"]], on="id", how="left")
    parts = {name: labelled[labelled["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")}
    parts["train_val"] = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    classes = sorted(split["cuisine"].unique())
    baskets = recipes.loc[~recipes["id"].isin(parts["test"]["id"]), "ingredients_v2"]
    frequency = Counter(i for b in baskets for i in set(b))
    print(f"{len(split):,} labelled recipes, {len(classes)} cuisines; graph built from {len(baskets):,} recipes "
          f"(all but test) ({time.perf_counter() - t0:.0f} s)")

    # ---- validation sweep ----------------------------------------------------------------------
    rows, partitions = [], {}
    for w in WEIGHTINGS:
        G = graphs.largest_component(cuisine.build_graph(baskets, weight=w))
        for res in RESOLUTIONS:
            part = graphs.louvain(G, seed=graphs.SEED, resolution=res)
            partitions[(w, res)] = (G, part)
            X = profiles(part, {"train": parts["train"], "val": parts["val"]})
            coverage = float((X["val"].sum(axis=1) > 0).mean())
            for m in METHODS:
                pred = fit_predict(m, X["train"], parts["train"]["cuisine"].to_numpy(), X["val"], classes)
                rows.append({"weighting": w, "resolution": res, "n_communities": max(part.values()) + 1,
                             "nodes": G.number_of_nodes(), "method": m, "coverage": coverage,
                             **cuisine.evaluate(parts["val"]["cuisine"], pred, classes)})
            best_here = max(r["macro_f1"] for r in rows[-3:])
            print(f"{w:>5} resolution {res:>4}: {max(part.values()) + 1:>3} communities, best val macro-F1 "
                  f"{best_here:.3f} ({time.perf_counter() - t0:.0f} s)")
    val = pd.DataFrame(rows)
    majority = classes[int(np.argmax(cuisine.class_prior(parts["train"]["cuisine"].to_numpy(), classes)))]
    refs_val = {
        "majority class": cuisine.evaluate(parts["val"]["cuisine"], [majority] * len(parts["val"]), classes)["macro_f1"],
        "logistic on all ingredients": cuisine.evaluate(
            parts["val"]["cuisine"], onehot_reference(parts["train"], parts["val"], classes), classes)["macro_f1"],
    }
    val.to_csv(RESULTS / "cuisine_val.csv", index=False, float_format="%.4f")
    pd.DataFrame({"reference": list(refs_val), "macro_f1": list(refs_val.values())}).to_csv(
        RESULTS / "cuisine_val_references.csv", index=False, float_format="%.4f")
    plot_resolution(val, refs_val, FIGURES / "cuisine_resolution.png")

    # ---- test: count vs PMI at the same number of communities, refit on train + val, 5 seeds ---------
    method = val.groupby("method")["macro_f1"].mean().idxmax()  # best predictor on validation
    test_rows, y_test, y_fit = [], parts["test"]["cuisine"].to_numpy(), parts["train_val"]["cuisine"].to_numpy()
    best_preds, kept = {}, {}
    for k_target in TARGET_K:
        for w in WEIGHTINGS:
            G = partitions[(w, RESOLUTIONS[0])][0]
            res, _ = resolution_for_k(G, k_target)
            scores = []
            for seed in SEEDS:
                part = graphs.louvain(G, seed=seed, resolution=res)
                X = profiles(part, {"train_val": parts["train_val"], "test": parts["test"]})
                pred = fit_predict(method, X["train_val"], y_fit, X["test"], classes)
                scores.append(cuisine.evaluate(y_test, pred, classes) | {"n_communities": max(part.values()) + 1})
                if seed == graphs.SEED:
                    kept[(w, k_target)] = (part, pred, cuisine.community_cuisine_table(X["train_val"], y_fit, classes))
            test_rows.append({"model": f"{w} graph communities", "method": method, "target_k": k_target,
                              "resolution": res, "n_communities": np.mean([s["n_communities"] for s in scores]),
                              "accuracy": np.mean([s["accuracy"] for s in scores]),
                              "accuracy_std": np.std([s["accuracy"] for s in scores]),
                              "macro_f1": np.mean([s["macro_f1"] for s in scores]),
                              "macro_f1_std": np.std([s["macro_f1"] for s in scores])})
            print(f"test, ~{k_target} communities, {w}: macro-F1 {test_rows[-1]['macro_f1']:.3f} "
                  f"(resolution {res:.2f}) ({time.perf_counter() - t0:.0f} s)")
    best_preds["pmi"], best_preds["count"] = kept[("pmi", TARGET_K[-1])][1], kept[("count", TARGET_K[-1])][1]
    pmi_part, _, pmi_table = kept[("pmi", TARGET_K[0])]          # ~10 communities: readable table
    best_preds["majority"] = np.array([majority] * len(y_test))
    best_preds["onehot"] = onehot_reference(parts["train_val"], parts["test"], classes)
    for name, key in (("majority class", "majority"), ("logistic on all ingredients", "onehot")):
        test_rows.append({"model": name, "method": "-", **cuisine.evaluate(y_test, best_preds[key], classes)})
    test = pd.DataFrame(test_rows)
    test.to_csv(RESULTS / "cuisine_test.csv", index=False, float_format="%.4f")

    # ---- details for the report -----------------------------------------------------------------------
    per_class = pd.DataFrame({"cuisine": classes, "n_test": [int((y_test == c).sum()) for c in classes]})
    for name, key in (("f1_pmi_graph_100", "pmi"), ("f1_count_graph_100", "count"), ("f1_onehot_logistic", "onehot")):
        per_class[name] = f1_score(y_test, best_preds[key], labels=classes, average=None, zero_division=0)
    per_class.to_csv(RESULTS / "cuisine_per_class.csv", index=False, float_format="%.4f")
    cm = confusion_matrix(y_test, best_preds["pmi"], labels=classes)  # PMI graph, ~100 communities
    pd.DataFrame(cm, index=classes, columns=classes).rename_axis("true \\ predicted").to_csv(RESULTS / "cuisine_confusion.csv")
    comm_rows = []
    for c in range(max(pmi_part.values()) + 1):
        members = sorted((n for n, k in pmi_part.items() if k == c), key=lambda n: -frequency[n])
        order = np.argsort(-pmi_table[c])
        comm_rows.append({"community": c, "size": len(members), "top_ingredients": ", ".join(members[:10]),
                          "top_cuisine": classes[order[0]], "top_share": pmi_table[c, order[0]],
                          "second_cuisine": classes[order[1]], "second_share": pmi_table[c, order[1]]})
    pd.DataFrame(comm_rows).to_csv(RESULTS / "cuisine_communities.csv", index=False, float_format="%.3f")

    print(val.sort_values("macro_f1", ascending=False).head(6).round(3).to_string(index=False))
    print(f"validation references: " + ", ".join(f"{k} {v:.3f}" for k, v in refs_val.items()))
    print(test.round(3).to_string(index=False))
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
