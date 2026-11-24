"""Predict a recipe's cuisine from the communities of the ingredient graph (R4).

Idea: Louvain groups ingredients that are used together (without looking at any label). A recipe is then
described by the share of its ingredients that falls in each community, e.g. 60 % "Mediterranean",
30 % "baking", 10 % "Mexican": a few numbers instead of thousands of ingredient columns. Two predictors:
  * without training a model: each community gets the cuisine distribution of the training recipes that
    use it, and a recipe takes the cuisine with the highest score (optionally divided by the cuisine's
    overall share, so frequent cuisines do not win everything);
  * a logistic regression on the community profile.
"""
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

import graphs


def build_graph(baskets: Iterable, weight: str = "pmi", min_support: float = 0.001,
                min_lcc_share: float = 0.9) -> nx.Graph:
    """Ingredient graph from baskets: frequent pairs (support >= min_support), weighted by `weight`,
    threshold = largest decile that keeps >= min_lcc_share of the ingredients connected
    (same rule as run_graph.py)."""
    pairs = graphs.pmi_weights(graphs.count_pairs(baskets, min_support))
    col = "pair_count" if weight == "count" else weight
    values = pairs[col] if weight == "count" else pairs.loc[pairs["pmi"] > 0, col]
    thresholds = np.unique(np.quantile(values, np.linspace(0, 0.9, 19))) if len(values) else [0.0]
    sweep = graphs.threshold_sweep(pairs, weight, thresholds)
    return graphs.ingredient_graph(pairs, weight, min_weight=graphs.choose_threshold(sweep, min_lcc_share))


def community_profiles(ingredient_lists: Iterable, partition: dict, n_communities: int) -> np.ndarray:
    """(n_recipes, n_communities): share of each recipe's distinct graph ingredients in each community.
    Ingredients that are not in the graph are ignored; a recipe without any gets a row of zeros."""
    rows = []
    for ingredients in ingredient_lists:
        row = np.zeros(n_communities)
        known = [partition[i] for i in set(ingredients) if i in partition]
        for c in known:
            row[c] += 1
        rows.append(row / len(known) if known else row)
    return np.vstack(rows) if rows else np.zeros((0, n_communities))


def community_cuisine_table(X: np.ndarray, y: np.ndarray, classes: list, alpha: float = 1.0) -> np.ndarray:
    """(n_communities, n_classes): P(cuisine | community), from the training profiles X and labels y,
    each recipe counting with the share of its ingredients in the community. Additive smoothing alpha;
    a community with no training weight gets a uniform row."""
    Y = np.stack([(np.asarray(y) == c).astype(float) for c in classes], axis=1)  # one-hot labels
    M = X.T @ Y + alpha
    totals = M.sum(axis=1, keepdims=True)
    uniform = np.full_like(M, 1.0 / len(classes))
    return np.divide(M, totals, out=uniform, where=totals > 0)


def predict_from_table(X: np.ndarray, table: np.ndarray, classes: list, prior: np.ndarray | None = None,
                       fallback: str | None = None) -> np.ndarray:
    """Cuisine with the highest score sum_k profile_k * P(cuisine | k), divided by prior(cuisine) if given.
    Recipes with an empty profile get `fallback` (default: most frequent class of the prior, else classes[0])."""
    scores = X @ (table / prior if prior is not None else table)
    pred = np.asarray(classes, dtype=object)[scores.argmax(axis=1)]
    if fallback is None:
        fallback = classes[int(np.argmax(prior))] if prior is not None else classes[0]
    pred[X.sum(axis=1) == 0] = fallback
    return pred


def evaluate(y_true, y_pred, classes: list) -> dict:
    """accuracy and macro-F1 (each cuisine counts the same, whatever its size)."""
    from sklearn.metrics import accuracy_score, f1_score

    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))}


def class_prior(y: np.ndarray, classes: list) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    return np.array([counts.get(c, 0) for c in classes], dtype=float) / len(y)
