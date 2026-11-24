"""Verify exact Jaccard, reproducible search, and the RQ2 benchmark outputs."""

from itertools import combinations

import pandas as pd
import pytest

from scripts.benchmark_recipe_similarity import (
    CONFIGURATIONS,
    THRESHOLDS,
    evaluate_candidates,
    main as benchmark_main,
)
from scripts.rq2_similarity import LSHIndex, RecipeCorpus, RecipeMatch, jaccard


@pytest.mark.parametrize(
    ("left", "right", "score"),
    [
        ({"egg", "milk"}, {"egg", "milk"}, 1.0),
        ({"egg"}, {"milk"}, 0.0),
        (set(), {"milk"}, 0.0),
        (set(), set(), 0.0),
        ({"egg", "milk"}, {"milk", "flour"}, 1 / 3),
    ],
)
def test_jaccard(left, right, score):
    assert jaccard(left, right) == pytest.approx(score)


def test_empty_queries_and_recipes_do_not_produce_matches():
    corpus = RecipeCorpus({1: [], 2: [" ", ""], 3: ["salt"]})
    index = LSHIndex(corpus)
    assert index.query_ingredients([]) == index.query_by_id(1) == []
    assert index.candidate_positions(1) == set()
    assert index.query_ingredients(["salt"], k=0) == []
    assert LSHIndex(RecipeCorpus({})).query_ingredients(["salt"]) == []


def test_identical_neighbors_use_numeric_then_text_id_ties_regardless_of_input_order():
    recipes = {100: ["EGG", "oil"], 10: ["egg", "oil"], 2: ["oil", " egg ", "egg"], "a": ["oil", "egg"]}
    expected = [RecipeMatch(2, 1.0), RecipeMatch(10, 1.0), RecipeMatch("a", 1.0)]
    index = LSHIndex(RecipeCorpus(recipes), seed=7)
    reordered = LSHIndex(RecipeCorpus(dict(reversed(recipes.items()))), seed=7)
    assert index.query_by_id(100, k=3) == reordered.query_by_id(100, k=3) == expected


def test_queries_find_true_nearest_neighbor_and_rerank_with_unseen_ingredients():
    corpus = RecipeCorpus({
        1: ["garlic", "oil", "tomato", "pasta"],
        2: ["garlic", "oil", "tomato"],
        3: ["garlic", "oil"],
        4: ["milk", "sugar"],
    })
    index = LSHIndex(corpus, bands=64, rows=2, seed=42)
    assert index.query_by_id(1, k=1) == corpus.exact_by_id(1, k=1) == [RecipeMatch(2, 0.75)]
    assert index.query_ingredients([" GARLIC ", "oil", "tomato", "unseen"], k=1) == [
        RecipeMatch(2, 0.75),
    ]
    with pytest.raises(KeyError):
        index.query_by_id(999)


def test_invalid_banding_is_rejected():
    with pytest.raises(ValueError, match="multiply"):
        LSHIndex(RecipeCorpus({1: ["salt"]}), bands=5, rows=5)


def test_candidate_metrics_match_independent_exact_jaccard_with_no_double_counting():
    corpus = RecipeCorpus({
        1: ["a", "b", "c", "d"],
        2: ["a", "b", "c", "d"],
        3: ["a", "b", "c"],
        4: ["z"],
        5: [],
    })
    indexes = [LSHIndex(corpus, bands=bands, rows=rows) for bands, rows in CONFIGURATIONS]
    measured = evaluate_candidates(corpus, indexes)
    for index in indexes:
        candidates = {
            (i, j)
            for i in range(len(corpus.recipe_ids))
            for j in index.candidate_positions(i)
            if i < j
        }
        for threshold in THRESHOLDS:
            positives = {
                (i, j)
                for i, j in combinations(range(len(corpus.recipe_ids)), 2)
                if jaccard(corpus.ingredient_sets[i], corpus.ingredient_sets[j]) >= threshold
            }
            row = measured.loc[
                (measured["bands"] == index.bands) & (measured["threshold"] == threshold)
            ].iloc[0]
            hits = len(candidates & positives)
            assert row["candidates"] == len(candidates)
            assert row["ground_truth_pairs"] == len(positives)
            assert row["true_positives"] == hits
            assert row["precision"] == pytest.approx(hits / len(candidates))
            assert row["recall"] == pytest.approx(hits / len(positives))
            assert row["f1"] == pytest.approx(2 * hits / (len(candidates) + len(positives)))


def test_no_exact_matches_or_candidates_yields_finite_zero_metrics():
    corpus = RecipeCorpus({1: ["a"], 2: ["b"], 3: []})
    metrics = evaluate_candidates(corpus, [LSHIndex(corpus)], thresholds=(1.0,))
    row = metrics.iloc[0]
    assert row["candidates"] == row["ground_truth_pairs"] == row["true_positives"] == 0
    assert row["precision"] == row["recall"] == row["f1"] == 0.0


def test_benchmark_cli_sweeps_all_configs_and_uses_paired_queries(tmp_path):
    ingredients = {
        21: ["pasta", "oil", "tomato"],
        22: ["pasta", "oil", "tomato"],
        23: ["pasta", "oil", "basil"],
        24: ["egg", "flour"],
        25: ["egg", "flour"],
        26: ["milk", "sugar"],
    }
    names = {recipe: f"Recipe {recipe}" for recipe in ingredients}
    recipes = tmp_path / "recipes.csv"
    pd.DataFrame({
        "recipe_id": list(ingredients), "name": list(names.values()), "ingredients_v2": list(ingredients.values()),
    }).to_csv(recipes, index=False)
    output = tmp_path / "lsh"
    benchmark_main([
        "--recipes", str(recipes), "--sample-size", "6", "--seed", "17", "--out-dir", str(output),
    ])

    metrics = pd.read_csv(output / "lsh_eval.csv")
    assert set(metrics[["bands", "rows", "threshold"]].itertuples(index=False, name=None)) == {
        (bands, rows, threshold) for bands, rows in CONFIGURATIONS for threshold in THRESHOLDS
    }
    assert metrics["sample_size"].eq(6).all()
    assert metrics["n_hashes"].eq(128).all()
    assert metrics.loc[metrics["threshold"] == 0.9, "ground_truth_pairs"].eq(2).all()

    latency = pd.read_csv(output / "search_latency.csv")
    exact = latency.loc[latency["method"] == "exact"]
    assert len(exact) == 6 and exact["index_build_sec"].eq(0).all()
    assert len(latency) == 6 * (1 + len(CONFIGURATIONS))
    for bands, rows in CONFIGURATIONS:
        group = latency.loc[(latency["bands"] == bands) & (latency["rows"] == rows)]
        assert group["query_recipe_id"].tolist() == exact["query_recipe_id"].tolist()
        assert group["index_build_sec"].gt(0).all()
    assert latency["ms_per_query"].ge(0).all()

    examples = pd.read_csv(output / "similar_examples.csv")
    assert examples["query_recipe_id"].nunique() == 3
    assert set(examples["method"]) == {"lsh", "exact"}
    assert all(examples["query_name"] == examples["query_recipe_id"].map(names))
    for row in examples.dropna(subset=["matched_recipe_id"]).itertuples():
        assert row.query_recipe_id != row.matched_recipe_id
        assert row.jaccard == pytest.approx(
            jaccard(set(ingredients[row.query_recipe_id]), set(ingredients[row.matched_recipe_id])),
        )
