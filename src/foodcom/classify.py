"""RQ4 – calorie level predicted from LSH neighbours (owner: R5, docs/CONTRACT.md 3.4.5).

Outside the course: k-nearest-neighbour classification, using the neighbours found by R2's LSH
(config.NEIGHBOURS). Label: recipes.calorie_level (0 low, 1 medium, 2 high, from R1).
No leakage: neighbours are restricted to training recipes, and near-duplicate groups
(config.DUPLICATES) are never split between train and test.
"""
import numpy as np
import pandas as pd

from foodcom import config


def train_test_split_recipes(recipes: pd.DataFrame, duplicates: pd.DataFrame | None = None,
                             test_frac: float = 0.2, seed: int = config.SEED) -> tuple[pd.Index, pd.Index]:
    """recipe_ids of the train and test sets (recipes with a calorie_level only). A recipe and its
    canonical copy (duplicates.canonical_id) always fall in the same set."""
    raise NotImplementedError("R5")


def knn_predict(neighbours: pd.DataFrame, labels: pd.Series, test_ids: pd.Index, train_ids: pd.Index,
                k: int = 10, weighted: bool = True) -> pd.Series:
    """Predicted calorie_level for each test recipe: vote of its k best neighbours that are in the
    training set (weighted by Jaccard similarity if weighted). labels: recipe_id -> calorie_level.
    Recipes with no training neighbour get <NA> (reported as coverage, never filled silently)."""
    raise NotImplementedError("R5")


def baseline_majority(labels: pd.Series, train_ids: pd.Index, test_ids: pd.Index) -> pd.Series:
    """Most frequent training label for every test recipe."""
    raise NotImplementedError("R5")


def baseline_logistic(recipes: pd.DataFrame, train_ids: pd.Index, test_ids: pd.Index,
                      seed: int = config.SEED) -> pd.Series:
    """Supervised reference: logistic regression (scikit-learn) on one-hot ingredients_norm."""
    raise NotImplementedError("R5")


def evaluate_classifier(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """{"coverage", "accuracy", "macro_f1", "confusion"} on the recipes with a prediction;
    coverage = share of test recipes that got one."""
    raise NotImplementedError("R5")
