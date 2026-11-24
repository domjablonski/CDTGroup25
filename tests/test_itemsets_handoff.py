"""Known counts and CLI file contracts for RQ1 and the RQ3 handoff."""

import json
from itertools import combinations

import pandas as pd
import pytest

from scripts.benchmark_itemsets import (
    BENCH_COLUMNS,
    apriori,
    encode_baskets,
    main as benchmark_main,
    run_pcy,
    support_count,
)
from scripts.export_pcy_pairs import PAIR_COLUMNS, main as export_main, validate_pairs

TOY_BASKETS = [
    ["a", "b", "b", "c"], ["a", "b"], ["a", "c"], ["a", "b", "d"],
    ["b", "c"], ["a", "b", "c"], ["d"], ["a", "e"],
]


def _recipes():
    return pd.DataFrame({"id": range(101, 109), "ingredients_v2": TOY_BASKETS})


@pytest.mark.parametrize("buckets", [1, 7, 1_024])
def test_pcy_matches_apriori_including_triples_and_bucket_collisions(buckets):
    baskets = encode_baskets(TOY_BASKETS)
    baseline = apriori(baskets, minimum=2, include_triples=True)
    pcy = run_pcy(baskets, minimum=2, buckets=buckets, include_triples=True)
    assert baseline.singles == pcy.singles == {0: 6, 1: 5, 2: 4, 3: 2}
    assert baseline.pairs == pcy.pairs == {(0, 1): 4, (0, 2): 3, (1, 2): 3}
    assert baseline.triples == pcy.triples == {(0, 1, 2): 2}
    assert baseline.candidate_pairs_counted == 5
    assert 3 <= pcy.candidate_pairs_counted <= 5
    if buckets == 1:
        assert pcy.candidate_pairs_counted == 5
    if buckets == 1_024:
        assert pcy.candidate_pairs_counted == 3


def test_support_ceil_includes_boundary_for_both_algorithms():
    baskets = encode_baskets(iter(basket) for basket in TOY_BASKETS)
    assert support_count(0.25, 8) == 2
    assert support_count(0.251, 8) == 3
    assert support_count(0.07, 100) == 7
    for minimum in (2, 3):
        baseline = apriori(baskets, minimum, include_triples=True)
        pcy = run_pcy(baskets, minimum, 1, include_triples=True)
        assert ((0, 1, 2) in baseline.triples) is (minimum == 2)
        assert baseline.triples == pcy.triples


def test_benchmark_cli_writes_counts_metrics_and_association_rules(tmp_path):
    recipes = tmp_path / "recipes.csv"
    _recipes().to_csv(recipes, index=False)
    output = tmp_path / "benchmark"
    benchmark_main([
        "--recipes", str(recipes), "--supports", "0.25", "--buckets", "1,1024",
        "--sample-size", "8", "--triples", "--out-dir", str(output),
    ])
    timing = pd.read_csv(output / "itemsets_bench.csv")
    assert list(timing) == BENCH_COLUMNS
    assert timing["method"].tolist() == ["apriori", "pcy", "pcy"]
    assert timing["support_count"].tolist() == [2, 2, 2]
    assert timing["frequent_singles"].tolist() == [4, 4, 4]
    assert timing["frequent_pairs"].tolist() == [3, 3, 3]
    assert timing["candidate_pairs_counted"].tolist() == [5, 5, 3]
    assert timing["exact_match_with_apriori"].tolist() == [True, True, True]
    assert (timing[["elapsed_sec", "peak_memory_mb"]] >= 0).all().all()

    rules = pd.read_csv(output / "rules_top.csv")
    assert len(rules) == 6
    first = rules.query("antecedent == 'a' and consequent == 'b'").iloc[0]
    reverse = rules.query("antecedent == 'b' and consequent == 'a'").iloc[0]
    assert first["support"] == reverse["support"] == pytest.approx(4 / 8)
    assert first["confidence"] == pytest.approx(4 / 6)
    assert reverse["confidence"] == pytest.approx(4 / 5)
    assert first["lift"] == reverse["lift"] == pytest.approx(4 * 8 / (6 * 5))


def test_export_uses_graph_adapter_schema_and_exact_counts(tmp_path):
    recipes = tmp_path / "recipes.parquet"
    _recipes().to_parquet(recipes, index=False)
    output = tmp_path / "handoff"
    export_main([
        "--recipes", str(recipes), "--support", "0.25", "--buckets", "1", "--out-dir", str(output),
    ])
    singles = pd.read_parquet(output / "frequent_ingredients_singles.parquet")
    graph_pairs = pd.read_parquet(output / "frequent_ingredients_pairs.parquet")
    pair_table = pd.read_parquet(output / "pairs.parquet")
    manifest = json.loads((output / "pairs_metadata.json").read_text(encoding="utf-8"))

    assert list(singles) == ["ingredient", "support"]
    assert list(graph_pairs) == ["ingredient1", "ingredient2", "support"]
    assert list(pair_table) == ["item_a", "item_b", "pair_count", "count_a", "count_b", "n_baskets"]
    assert manifest["n_baskets"] == 8
    assert manifest["vocabulary"] == ["a", "b", "c", "d", "e"]
    assert manifest["support_count"] == 2
    assert "--recipes" in manifest["command"] and str(recipes) in manifest["command"]
    sizes = singles.set_index("ingredient")["support"].to_dict()
    assert sizes == {"a": 6, "b": 5, "c": 4, "d": 2}
    expected = []
    for a, b in combinations(sorted(sizes), 2):
        count = sum({a, b}.issubset(basket) for basket in TOY_BASKETS)
        if count >= manifest["support_count"]:
            expected.append((a, b, count, sizes[a], sizes[b], manifest["n_baskets"]))
    assert list(graph_pairs.itertuples(index=False, name=None)) == [row[:3] for row in expected]
    assert list(pair_table.itertuples(index=False, name=None)) == expected


def test_export_drops_copies_preserves_canonical_ids_and_recomputes_denominator(tmp_path):
    recipes = tmp_path / "recipes.csv"
    pd.DataFrame({
        "id": [10, 11, 12, 13],
        "cleaned": [["garlic", "oil"], ["garlic", "oil", "pepper"], ["oil", "salt"], ["garlic", "oil"]],
    }).to_csv(recipes, index=False)
    duplicates = tmp_path / "duplicates.parquet"
    pd.DataFrame({"recipe_id": [10, 11, 13], "canonical_id": [10, 10, 10]}).to_parquet(duplicates, index=False)
    output = tmp_path / "handoff"
    export_main([
        "--recipes", str(recipes), "--ingredient-col", "cleaned", "--support", "0.5",
        "--duplicates", str(duplicates), "--out-dir", str(output),
    ])
    pair_table = pd.read_parquet(output / "pairs.parquet")
    assert list(pair_table.itertuples(index=False, name=None)) == [
        ("garlic", "oil", 1, 1, 2, 2), ("oil", "salt", 1, 2, 1, 2),
    ]
    manifest = json.loads((output / "pairs_metadata.json").read_text(encoding="utf-8"))
    assert manifest["duplicates_removed"] == 2
    assert manifest["n_baskets"] == 2
    assert manifest["ingredient_col"] == "cleaned"
    assert manifest["vocabulary"] == ["garlic", "oil", "salt"]


def test_empty_pair_export_retains_schema(tmp_path):
    recipes = tmp_path / "recipes.csv"
    _recipes().to_csv(recipes, index=False)
    output = tmp_path / "handoff"
    export_main(["--recipes", str(recipes), "--support", "1", "--out-dir", str(output)])
    pairs = pd.read_parquet(output / "pairs.parquet")
    graph_pairs = pd.read_parquet(output / "frequent_ingredients_pairs.parquet")
    assert pairs.empty and list(pairs) == PAIR_COLUMNS
    assert graph_pairs.empty and list(graph_pairs) == ["ingredient1", "ingredient2", "support"]
    validate_pairs(pairs, n_baskets=8, minimum=8)


def test_schema_validation_rejects_inconsistent_pair_counts():
    pairs = pd.DataFrame([("a", "b", 2, 1, 2, 4)], columns=PAIR_COLUMNS)
    with pytest.raises(ValueError, match="counts"):
        validate_pairs(pairs, n_baskets=4, minimum=1)
