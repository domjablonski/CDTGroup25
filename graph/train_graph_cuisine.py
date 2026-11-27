"""Exploration (VALIDATION ONLY): train the PMI graph on the annotated recipes to predict cuisine better. (R4)

The graph never changes: the frequent pairs at support 0.1 % on all recipes (the PCY pairs agreed with R3),
PMI weights, same threshold rule. Only the cuisine labels of the TRAINING recipes are used, to:
  A. baseline: Louvain communities (~100) -> profile -> logistic regression (current method)
  B. same graph, soft membership + 10/30/100 communities together
  C. label-guided communities: an ingredient's cuisine distribution comes from the training recipes that
     contain it; each edge weight is multiplied by the similarity of the two ingredients' distributions
     (cosine, squared), then Louvain -> communities that follow cuisines more closely
  D. cuisine propagation: the ingredients' cuisine distributions are smoothed over the PMI graph
     (P = (1 - b) P0 + b W P), and a recipe gets the mean of its ingredients' distributions
  Control: beta = 0 uses each ingredient's cuisine distribution WITHOUT the graph. If it does as well as
  propagation, the gain comes from the labels, not from the graph.
Labels of a training recipe never feed a feature of that same recipe when the logistic regression is fitted
(5-fold out-of-fold distributions), so validation scores are honest.

Usage (repo root): python graph/train_graph_cuisine.py -> graph/results/cuisine_trained_val.csv
"""
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import cuisine
import graphs
from run_cuisine import SPLIT, resolution_for_k

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "cuisine_trained_val.csv"
C_VALUES = [0.3, 1.0, 3.0]


def best_logistic(X_tr, y_tr, X_va, y_va, classes):
    scores = []
    for C in C_VALUES:
        pred = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X_tr, y_tr).predict(X_va)
        scores.append((cuisine.evaluate(y_va, pred, classes)["macro_f1"], C))
    return max(scores)


def ingredient_cuisine(R: sp.csr_matrix, y, classes, prior, alpha=5.0) -> np.ndarray:
    """(n_ingredients, n_classes): share of each cuisine among the recipes containing the ingredient,
    smoothed towards the overall prior with alpha pseudo-recipes."""
    Y = np.stack([(np.asarray(y) == c).astype(float) for c in classes], axis=1)
    counts = R.T @ Y
    return (counts + alpha * prior) / (counts.sum(axis=1, keepdims=True) + alpha)


def propagate(P0: np.ndarray, W: sp.csr_matrix, beta=0.5, steps=20) -> np.ndarray:
    P = P0.copy()
    for _ in range(steps):
        P = (1 - beta) * P0 + beta * (W @ P)
    return P


def main() -> None:
    t0 = time.perf_counter()
    recipes = graphs.load_recipes()
    split = pd.read_csv(SPLIT).merge(recipes[["id", "ingredients_v2"]], on="id")
    train, val = split[split["split"] == "train"].reset_index(drop=True), split[split["split"] == "val"]
    classes = sorted(split["cuisine"].unique())
    y_tr, y_va = train["cuisine"].to_numpy(), val["cuisine"].to_numpy()
    prior = cuisine.class_prior(y_tr, classes)

    # the agreed graph: PCY pairs at 0.1 % on ALL recipes, PMI, same threshold rule
    G = graphs.largest_component(cuisine.build_graph(recipes["ingredients_v2"], weight="pmi", min_support=0.001))
    nodes = sorted(G)
    index = {n: i for i, n in enumerate(nodes)}
    R_tr, R_va = cuisine.incidence(train["ingredients_v2"].tolist(), index), cuisine.incidence(val["ingredients_v2"].tolist(), index)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    W = sp.diags(1 / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-12)) @ A  # row-normalised PMI weights
    print(f"graph: {len(nodes)} ingredients, {G.number_of_edges():,} edges ({time.perf_counter() - t0:.0f} s)")
    rows = []

    def record(name, f1, C=None, n_features=None):
        rows.append({"method": name, "macro_f1": f1, "C": C, "n_features": n_features})
        print(f"{name:<45} val macro-F1 {f1:.3f} ({time.perf_counter() - t0:.0f} s)")

    # A and B: unsupervised communities on the agreed graph
    memberships = {}
    for k in (10, 30, 100):
        res, _ = resolution_for_k(G, k, grid=15, steps=10)
        memberships[k] = cuisine.hard_membership(graphs.louvain(G, resolution=res), nodes)
    f1, C = best_logistic(cuisine.recipe_features(R_tr, memberships[100]), y_tr, cuisine.recipe_features(R_va, memberships[100]), y_va, classes)
    record("A  communities (100), logistic", f1, C, 100)
    soft = [cuisine.soft_membership(G, nodes, memberships[k]) for k in (10, 30, 100)]
    X_tr, X_va = np.hstack([cuisine.recipe_features(R_tr, F) for F in soft]), np.hstack([cuisine.recipe_features(R_va, F) for F in soft])
    f1, C = best_logistic(X_tr, y_tr, X_va, y_va, classes)
    record("B  soft, 10+30+100 communities, logistic", f1, C, X_tr.shape[1])

    # C: label-guided communities (edge weight x cuisine similarity of the two ingredients)
    P0 = ingredient_cuisine(R_tr, y_tr, classes, prior)
    unit = P0 / np.linalg.norm(P0, axis=1, keepdims=True)
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, weight=d["weight"] * float(unit[index[u]] @ unit[index[v]]) ** 2)
    for k in (30, 100):
        res, k_got = resolution_for_k(H, k, grid=15, steps=10)
        M = cuisine.hard_membership(graphs.louvain(H, resolution=res), nodes)
        f1, C = best_logistic(cuisine.recipe_features(R_tr, M), y_tr, cuisine.recipe_features(R_va, M), y_va, classes)
        record(f"C  label-guided communities ({k_got}), logistic", f1, C, k_got)

    # D: cuisine propagation over the graph
    for beta in (0.0, 0.3, 0.5, 0.7):
        P = propagate(P0, W, beta)
        X_va = cuisine.recipe_features(R_va, P)
        pred = cuisine.predict_from_table(X_va, np.eye(len(classes)), classes, prior=prior)
        record(f"D  propagation beta={beta}, no model (mean / prior)", cuisine.evaluate(y_va, pred, classes)["macro_f1"])
    # D + logistic: out-of-fold ingredient distributions for the training recipes (no label leakage).
    # beta = 0 is the control WITHOUT the graph: only each ingredient's cuisine distribution is used.
    oof = {}
    for beta in (0.0, 0.3, 0.5, 0.7):
        X_tr = np.zeros((len(train), len(classes)))
        for fit_idx, oof_idx in KFold(5, shuffle=True, random_state=graphs.SEED).split(train):
            P_fold = propagate(ingredient_cuisine(R_tr[fit_idx], y_tr[fit_idx], classes, prior), W, beta)
            X_tr[oof_idx] = cuisine.recipe_features(R_tr[oof_idx], P_fold)
        X_va = cuisine.recipe_features(R_va, propagate(P0, W, beta))
        oof[beta] = (X_tr, X_va)
        f1, C = best_logistic(X_tr, y_tr, X_va, y_va, classes)
        label = " (control: no graph)" if beta == 0.0 else ""
        record(f"D  propagation beta={beta}, logistic{label}", f1, C, len(classes))
    X_tr, X_va = oof[0.5]
    # D + communities together
    X_tr2 = np.hstack([X_tr, cuisine.recipe_features(R_tr, memberships[100])])
    X_va2 = np.hstack([X_va, cuisine.recipe_features(R_va, memberships[100])])
    f1, C = best_logistic(X_tr2, y_tr, X_va2, y_va, classes)
    record("D+A propagation + communities (100), logistic", f1, C, X_tr2.shape[1])

    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.4f")
    print(f"done in {time.perf_counter() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
