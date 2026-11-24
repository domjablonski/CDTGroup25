"""Pair validation, grouped splitting, and merge metric checks."""

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_ingredient_merges import (
    LABEL,
    NAMES,
    SOURCE_COLUMNS,
    CutoffRule,
    PairSet,
    evaluate,
    grouped_split,
    ingredient_groups,
    main,
    merge_metrics,
    read_labelled_pairs,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _validation_rows():
    return [
        (f"ingredient {i:02d}", f"variant {i:02d}", 0.92 if i % 2 else 0.78,
         0.65 if i % 2 else 0.25, 0.70 if i % 2 else 0.30, i % 2)
        for i in range(16)
    ]


def _write_csv(path, rows):
    _frame(rows).to_csv(path, index=False)
    return path


def test_blank_rows_and_reversed_pairs_are_counted(tmp_path):
    path = _write_csv(tmp_path / "labels.csv", [
        (" Garlic ", " garlic  powder ", 0.92, 0.5, 0.7, 0),
        ("GARLIC POWDER", "garlic", 0.92, 0.5, 0.7, 0),
        ("oil", "", "", "", "", ""),
        ("bean", "vanilla bean", 0.8, 0.5, 0.4, 0),
    ])
    pairs = read_labelled_pairs(path)
    assert (pairs.source_rows, pairs.blank_rows, pairs.duplicate_rows) == (4, 1, 1)
    assert list(pairs.table[list(NAMES)].itertuples(index=False, name=None)) == [
        ("garlic", "garlic powder"), ("bean", "vanilla bean"),
    ]


@pytest.mark.parametrize("label", ["2", "maybe"])
def test_nonbinary_labels_raise(tmp_path, label):
    path = _write_csv(tmp_path / "labels.csv", [("garlic", "garlic powder", 0.92, 0.5, 0.7, label)])
    with pytest.raises(ValueError, match="labels outside"):
        read_labelled_pairs(path)


def test_conflicting_unordered_pair_labels_raise(tmp_path):
    path = _write_csv(tmp_path / "labels.csv", [
        ("b", "a", 0.9, 0.5, 0.6, 0), ("a", "b", 0.9, 0.5, 0.6, 1),
    ])
    with pytest.raises(ValueError, match="Conflicting labels"):
        read_labelled_pairs(path)


def test_out_of_range_similarity_raises(tmp_path):
    path = _write_csv(tmp_path / "labels.csv", [("a", "b", 0.9, 1.1, 0.6, 0)])
    with pytest.raises(ValueError, match="out-of-range token_jaccard"):
        read_labelled_pairs(path)


def test_transitive_name_links_group_validation_rows_together():
    validation = _frame(_validation_rows())
    exploratory = _frame([
        ("variant 00", "bridge", 0.9, 0.5, 0.6, 0),
        ("bridge", "ingredient 01", 0.9, 0.5, 0.6, 0),
    ])
    components = ingredient_groups(pd.concat([validation, exploratory]))
    assert len({components[name] for name in (
        "ingredient 00", "variant 00", "bridge", "ingredient 01", "variant 01",
    )}) == 1
    dev, holdout = grouped_split(validation, exploratory)
    assert set(dev["group"]).isdisjoint(holdout["group"])
    assert ({0, 1} <= set(dev.index)) ^ ({0, 1} <= set(holdout.index))
    dev_names = set(dev[NAMES[0]]) | set(dev[NAMES[1]])
    holdout_names = set(holdout[NAMES[0]]) | set(holdout[NAMES[1]])
    assert dev_names.isdisjoint(holdout_names)


def test_each_multiscore_rule_has_a_fixed_logical_condition():
    features = np.array([[0.9, 0.4, 0.8], [0.9, 0.6, 0.8], [0.7, 0.6, 0.8], [0.9, 0.4, 0.3]])
    assert CutoffRule("all_three", (0.8, 0.5, 0.5)).predict(features).tolist() == [False, True, False, False]
    assert CutoffRule("cosine_and_either", (0.8, 0.5, 0.5)).predict(features).tolist() == [
        True, True, False, False,
    ]


def test_exploratory_test_labels_do_not_change_development_or_holdout_metrics():
    validation = PairSet(_frame(_validation_rows()), 16, 0, 0)
    exploratory = _frame([
        ("garlic", "garlic powder", 0.94, 0.60, 0.70, 0),
        ("shallot", "whole shallot", 0.95, 0.55, 0.76, 1),
    ])
    first, _ = evaluate(validation, PairSet(exploratory, 2, 0, 0))
    flipped = exploratory.assign(merge_label=1 - exploratory[LABEL])
    second, _ = evaluate(validation, PairSet(flipped, 2, 0, 0))
    held = first["split"] != "original_test_exploratory"
    pd.testing.assert_frame_equal(first.loc[held], second.loc[held])


def test_counts_define_precision_recall_and_f1():
    metrics = merge_metrics(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 0, 1, 0], dtype=bool))
    assert metrics["true_merges"] == 2 and metrics["false_merges"] == 1
    assert metrics["missed_merges"] == 1 and metrics["accepted_count"] == 3
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == pytest.approx(2 / 3)
    assert merge_metrics(np.array([0]), np.array([False]))["f1"] == 0.0


def test_cli_writes_audit_metrics_and_real_false_positive_rows(tmp_path):
    training = _validation_rows()
    validation = _write_csv(tmp_path / "validation.csv", training)
    test = _write_csv(tmp_path / "test.csv", [
        training[0], ("garlic", "garlic powder", 0.98, 0.85, 0.90, 0),
        ("shallot", "whole shallot", 0.95, 0.55, 0.76, 1), ("", "", "", "", "", ""),
    ])
    output = tmp_path / "results"
    main(["--val-csv", str(validation), "--test-csv", str(test), "--out-dir", str(output)])
    metrics = pd.read_csv(output / "ingredient_merge_eval.csv")
    errors = pd.read_csv(output / "ingredient_merge_errors.csv")
    assert len(metrics) == 18
    assert metrics["original_shared_pairs"].eq(1).all()
    assert metrics["original_shared_ingredients"].eq(2).all()
    assert metrics["test_blank_rows"].eq(1).all()
    assert metrics["dev_holdout_shared_ingredients"].eq(0).all()
    assert metrics.loc[metrics["split"] == "original_test_exploratory", "n_pairs"].eq(2).all()
    assert (metrics.groupby("method")["settings"].nunique() == 1).all()
    assert (metrics.groupby("split")["selected_by_development"].sum() == 1).all()
    assert errors[LABEL].eq(0).all()
    assert ("garlic", "garlic powder") in set(errors[list(NAMES)].itertuples(index=False, name=None))
    expected = metrics.loc[metrics["split"] != "development"].set_index(["split", "method"])["false_merges"]
    observed = errors.groupby(["split", "method"]).size().reindex(expected.index, fill_value=0)
    assert observed.tolist() == expected.tolist()
