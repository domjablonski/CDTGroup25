"""Grouped evaluation of ingredient-name merge decisions."""

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .recipe_data import DEFAULT_SEED, staged_outputs
else:
    from recipe_data import DEFAULT_SEED, staged_outputs

NAMES = ("ingredient_1", "ingredient_2")
SCORES = ("similarity", "token_jaccard", "char_trigram")
LABEL = "merge_label"
SOURCE_COLUMNS = (*NAMES, *SCORES, LABEL)
SINGLE_METHODS = {"cosine_cutoff": 0, "token_cutoff": 1, "trigram_cutoff": 2}
COSINE_CUTOFFS = (0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)
LEXICAL_CUTOFFS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
CUTOFF_GRIDS = (COSINE_CUTOFFS, LEXICAL_CUTOFFS, LEXICAL_CUTOFFS)
HOLDOUT_FRACTION = 0.25
PROBABILITY_CUTOFF = 0.50


@dataclass(frozen=True)
class PairSet:
    table: pd.DataFrame
    source_rows: int
    blank_rows: int
    duplicate_rows: int


def _check_pair_labels(pairs: pd.DataFrame) -> None:
    conflicts = pairs.groupby(list(NAMES))[LABEL].nunique().gt(1)
    if conflicts.any():
        pair = conflicts[conflicts].index[0]
        raise ValueError(f"Conflicting labels for ingredient pair {pair!r}.")


def read_labelled_pairs(path: Path) -> PairSet:
    """Typed observations with normalized names and one row per unordered pair."""
    raw = pd.read_csv(
        path, usecols=list(SOURCE_COLUMNS), dtype="string", keep_default_na=False, skip_blank_lines=False,
    ).loc[:, list(SOURCE_COLUMNS)]
    raw = raw.apply(lambda col: col.str.strip())
    blank = raw.isna().any(axis=1) | raw.eq("").any(axis=1)
    pairs = raw.loc[~blank].copy()
    if not pairs[LABEL].isin(["0", "1"]).all():
        raise ValueError(f"{path} contains labels outside {{0, 1}}.")
    pairs[LABEL] = pairs[LABEL].astype("int8")
    for score in SCORES:
        pairs[score] = pd.to_numeric(pairs[score])
        low = -1.0 if score == "similarity" else 0.0
        if not pairs[score].between(low, 1.0).all():
            raise ValueError(f"{path} contains an out-of-range {score} value.")

    left, right = (pairs[name].str.split().str.join(" ").str.casefold() for name in NAMES)
    ordered = left <= right
    pairs[NAMES[0]] = left.where(ordered, right)
    pairs[NAMES[1]] = right.where(ordered, left)
    _check_pair_labels(pairs)
    unique = pairs.drop_duplicates(list(NAMES), ignore_index=True)
    return PairSet(unique, len(raw), int(blank.sum()), len(pairs) - len(unique))


def ingredient_groups(pairs: pd.DataFrame) -> dict[str, str]:
    """Connected components of the observed ingredient-pair graph."""
    parent: dict[str, str] = {}

    def root(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for left, right in pairs[list(NAMES)].itertuples(index=False, name=None):
        a, b = sorted((root(left), root(right)))
        parent[b] = a
    return {name: root(name) for name in parent}


def _ingredient_names(pairs: pd.DataFrame) -> set[str]:
    return set(pairs[NAMES[0]]) | set(pairs[NAMES[1]])


def grouped_split(validation: pd.DataFrame, exploratory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development and holdout drawn from validation, using both sets' name links."""
    components = ingredient_groups(pd.concat([validation, exploratory], ignore_index=True))
    validation = validation.assign(group=validation[NAMES[0]].map(components))
    if validation["group"].nunique() < 2 or validation[LABEL].nunique() != 2:
        raise ValueError("A grouped split requires at least two components and both labels.")
    splitter = GroupShuffleSplit(n_splits=64, test_size=HOLDOUT_FRACTION, random_state=DEFAULT_SEED)
    for dev_idx, holdout_idx in splitter.split(validation, groups=validation["group"]):
        dev, holdout = validation.iloc[dev_idx], validation.iloc[holdout_idx]
        if dev[LABEL].nunique() == holdout[LABEL].nunique() == 2:
            return dev, holdout
    raise ValueError("No grouped split had both labels in development and holdout.")


@dataclass(frozen=True)
class CutoffRule:
    method: str
    cutoffs: tuple[float, ...]

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.method in SINGLE_METHODS:
            return scores[:, SINGLE_METHODS[self.method]] >= self.cutoffs[0]
        passes = scores >= self.cutoffs
        if self.method == "all_three":
            return passes.all(axis=1)
        return passes[:, 0] & passes[:, 1:].any(axis=1)

    def settings(self) -> dict:
        if self.method in SINGLE_METHODS:
            return {SCORES[SINGLE_METHODS[self.method]]: self.cutoffs[0]}
        return dict(zip(SCORES, self.cutoffs))


def merge_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Counts and rates for a positive merge decision."""
    merged = predicted.astype(bool)
    equivalent = actual.astype(bool)
    true_merges = int(np.count_nonzero(merged & equivalent))
    false_merges = int(np.count_nonzero(merged & ~equivalent))
    missed_merges = int(np.count_nonzero(~merged & equivalent))
    accepted = true_merges + false_merges
    positives = true_merges + missed_merges
    return {
        "n_pairs": len(actual),
        "positive_pairs": positives,
        "accepted_count": accepted,
        "true_merges": true_merges,
        "false_merges": false_merges,
        "missed_merges": missed_merges,
        "precision": true_merges / accepted if accepted else 0.0,
        "recall": true_merges / positives if positives else 0.0,
        "f1": 2 * true_merges / (accepted + positives) if accepted + positives else 0.0,
    }


def _ranking(metrics: dict) -> tuple[float, float, int]:
    return metrics["f1"], metrics["precision"], -metrics["accepted_count"]


def _tune_rules(scores: np.ndarray, labels: np.ndarray) -> list[CutoffRule]:
    grids = {method: (CUTOFF_GRIDS[i],) for method, i in SINGLE_METHODS.items()}
    grids.update({"all_three": CUTOFF_GRIDS, "cosine_and_either": CUTOFF_GRIDS})
    rules = []
    for method, grids_for_rule in grids.items():
        candidates = (CutoffRule(method, cutoffs) for cutoffs in product(*grids_for_rule))
        rules.append(max(candidates, key=lambda rule: _ranking(merge_metrics(labels, rule.predict(scores)))))
    return rules


def evaluate(validation: PairSet, original_test: PairSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cutoff selection and model fitting confined to the development components."""
    val, test = validation.table, original_test.table
    _check_pair_labels(pd.concat([val, test], ignore_index=True))
    validation_keys = set(val[list(NAMES)].itertuples(index=False, name=None))
    shared = [pair in validation_keys for pair in test[list(NAMES)].itertuples(index=False, name=None)]
    exploratory = test.iloc[np.flatnonzero(np.logical_not(shared))]
    dev, holdout = grouped_split(val, exploratory)
    dev_scores, dev_labels = dev[list(SCORES)].to_numpy(), dev[LABEL].to_numpy()

    methods = [(rule.method, rule.settings(), rule.predict) for rule in _tune_rules(dev_scores, dev_labels)]
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1_000))
    logistic.fit(dev_scores, dev_labels)

    def logistic_prediction(scores: np.ndarray) -> np.ndarray:
        return logistic.predict_proba(scores)[:, 1] >= PROBABILITY_CUTOFF

    methods.append(("logistic_l2", {"C": 1.0, "probability_cutoff": PROBABILITY_CUTOFF}, logistic_prediction))
    chosen = max(methods, key=lambda method: _ranking(merge_metrics(dev_labels, method[2](dev_scores))))[0]
    diagnostics = {
        "validation_source_rows": validation.source_rows,
        "validation_blank_rows": validation.blank_rows,
        "validation_duplicate_rows": validation.duplicate_rows,
        "test_source_rows": original_test.source_rows,
        "test_blank_rows": original_test.blank_rows,
        "test_duplicate_rows": original_test.duplicate_rows,
        "original_shared_pairs": sum(shared),
        "original_shared_ingredients": len(_ingredient_names(val) & _ingredient_names(test)),
        "dev_holdout_shared_ingredients": len(_ingredient_names(dev) & _ingredient_names(holdout)),
    }

    metrics = []
    errors = []
    splits = {"development": dev, "grouped_holdout": holdout, "original_test_exploratory": exploratory}
    for split, pairs in splits.items():
        features, labels = pairs[list(SCORES)].to_numpy(), pairs[LABEL].to_numpy()
        for method, settings, predict in methods:
            predictions = predict(features) if len(pairs) else np.empty(0, dtype=bool)
            metrics.append({
                "split": split,
                "method": method,
                "settings": json.dumps(settings, sort_keys=True),
                "selected_by_development": method == chosen,
                "n_ingredients": len(_ingredient_names(pairs)),
                **merge_metrics(labels, predictions),
                **diagnostics,
            })
            if split != "development":
                false_merges = pairs.loc[predictions & (labels == 0), list(SOURCE_COLUMNS)]
                errors.extend({"split": split, "method": method, **row}
                              for row in false_merges.to_dict("records"))
    error_table = pd.DataFrame(errors, columns=["split", "method", *SOURCE_COLUMNS])
    error_table = error_table.sort_values(
        ["split", "method", "similarity", *NAMES], ascending=[True, True, False, True, True],
        ignore_index=True,
    )
    return pd.DataFrame(metrics), error_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    metrics, errors = evaluate(read_labelled_pairs(args.val_csv), read_labelled_pairs(args.test_csv))
    with staged_outputs(args.out_dir) as stage:
        metrics.to_csv(stage / "ingredient_merge_eval.csv", index=False)
        errors.to_csv(stage / "ingredient_merge_errors.csv", index=False)
    selected = metrics.loc[metrics["selected_by_development"], [
        "split", "method", "n_pairs", "precision", "recall", "f1", "accepted_count",
    ]]
    print(selected.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
