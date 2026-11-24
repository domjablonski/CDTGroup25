"""Export PCY ingredient counts for the graph branch's pairs_from_pcy adapter."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__:
    from .benchmark_itemsets import Baskets, Itemsets, encode_baskets, parse_support, run_pcy, support_count
    from .recipe_data import RecipeId, parse_recipe_id, read_recipes, read_table, recipe_ids, staged_outputs
else:
    from benchmark_itemsets import Baskets, Itemsets, encode_baskets, parse_support, run_pcy, support_count
    from recipe_data import RecipeId, parse_recipe_id, read_recipes, read_table, recipe_ids, staged_outputs

PAIR_COLUMNS = ["item_a", "item_b", "pair_count", "count_a", "count_b", "n_baskets"]
SINGLE_COLUMNS = ["ingredient", "support"]
GRAPH_PAIR_COLUMNS = ["ingredient1", "ingredient2", "support"]


def without_duplicates(recipes: pd.DataFrame, duplicates: pd.DataFrame) -> pd.DataFrame:
    """Remove explicitly mapped copies and keep their final canonical IDs."""
    canonical_by_id: dict[RecipeId, RecipeId] = {}
    for recipe, canonical in duplicates[["recipe_id", "canonical_id"]].itertuples(index=False, name=None):
        recipe = parse_recipe_id(recipe)
        canonical = parse_recipe_id(canonical)
        if canonical_by_id.setdefault(recipe, canonical) != canonical:
            raise ValueError(f"Recipe {recipe!r} has more than one canonical ID.")
    copies = {recipe for recipe, canonical in canonical_by_id.items() if recipe != canonical}
    if copies.intersection(canonical_by_id.values()):
        raise ValueError("Each canonical ID must be the final representative of its duplicate group.")
    return recipes.loc[[recipe not in copies for recipe in recipe_ids(recipes)]]


def validate_pairs(pairs: pd.DataFrame, n_baskets: int, minimum: int) -> None:
    """Validate column order, types, counts, and the one-row-per-pair invariant."""
    if list(pairs.columns) != PAIR_COLUMNS or pairs.isna().to_numpy().any():
        raise ValueError("Pair table has missing values or an unexpected schema.")
    if not all(pd.api.types.is_integer_dtype(pairs[col]) for col in PAIR_COLUMNS[2:]):
        raise ValueError("Pair and basket counts must be integers.")
    if not all(pd.api.types.is_string_dtype(pairs[col]) for col in PAIR_COLUMNS[:2]):
        raise ValueError("Ingredient labels must be strings.")
    if (pairs["item_a"] >= pairs["item_b"]).any() or pairs.duplicated(["item_a", "item_b"]).any():
        raise ValueError("Pairs must contain two sorted ingredients and occur only once.")
    valid_counts = (
        pairs["n_baskets"].eq(n_baskets)
        & pairs["pair_count"].ge(minimum)
        & pairs["pair_count"].le(pairs["count_a"])
        & pairs["pair_count"].le(pairs["count_b"])
        & pairs["count_a"].le(n_baskets)
        & pairs["count_b"].le(n_baskets)
    )
    if not valid_counts.all():
        raise ValueError("Pair counts, singleton counts, or basket totals are inconsistent.")


def pair_tables(
    baskets: Baskets, itemsets: Itemsets, minimum: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Andreas's singles/pairs files and the expanded Food.com pair table."""
    names = baskets.vocabulary
    n_baskets = len(baskets.transactions)
    singles = pd.DataFrame(
        [(names[item], count) for item, count in sorted(itemsets.singles.items())],
        columns=SINGLE_COLUMNS,
    ).astype({"ingredient": "string", "support": "int64"})
    pair_table = pd.DataFrame(
        [
            (names[a], names[b], count, itemsets.singles[a], itemsets.singles[b], n_baskets)
            for (a, b), count in sorted(itemsets.pairs.items())
        ],
        columns=PAIR_COLUMNS,
    ).astype({"item_a": "string", "item_b": "string", **dict.fromkeys(PAIR_COLUMNS[2:], "int64")})
    validate_pairs(pair_table, n_baskets, minimum)
    graph_pairs = pair_table[["item_a", "item_b", "pair_count"]].set_axis(GRAPH_PAIR_COLUMNS, axis=1)
    return singles, graph_pairs, pair_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="For graph/run_graph.py, use the same recipe rows used here; the adapter takes their row count.",
    )
    parser.add_argument("--recipes", type=Path, required=True, help="Recipe CSV or Parquet")
    parser.add_argument("--ingredient-col", default="ingredients_v2")
    parser.add_argument("--support", type=parse_support, default="0.001")
    parser.add_argument("--buckets", type=int, default=100_000)
    parser.add_argument("--duplicates", type=Path, help="CSV or Parquet with recipe_id and canonical_id")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/pcy_handoff"))
    args = parser.parse_args(argv)

    recipes = read_recipes(args.recipes, args.ingredient_col)
    original_count = len(recipes)
    if args.duplicates is not None:
        duplicates = read_table(args.duplicates, ("recipe_id", "canonical_id"))
        recipes = without_duplicates(recipes, duplicates)
    baskets = encode_baskets(recipes[args.ingredient_col])
    n_baskets = len(baskets.transactions)
    minimum = support_count(args.support, n_baskets)
    counts = run_pcy(baskets, minimum, args.buckets)
    singles, graph_pairs, pairs = pair_tables(baskets, counts, minimum)
    manifest = {
        "command": sys.orig_argv if argv is None else ["python", "-m", "scripts.export_pcy_pairs", *argv],
        "recipes": str(args.recipes.resolve()),
        "ingredient_col": args.ingredient_col,
        "duplicates": str(args.duplicates.resolve()) if args.duplicates is not None else None,
        "duplicates_removed": original_count - n_baskets,
        "n_baskets": n_baskets,
        "support": float(args.support),
        "support_count": minimum,
        "buckets": args.buckets,
        "vocabulary": baskets.vocabulary,
    }
    with staged_outputs(args.out_dir) as stage:
        singles.to_parquet(stage / "frequent_ingredients_singles.parquet", index=False)
        graph_pairs.to_parquet(stage / "frequent_ingredients_pairs.parquet", index=False)
        pairs.to_parquet(stage / "pairs.parquet", index=False)
        (stage / "pairs_metadata.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print(f"Exported {len(pairs):,} frequent pairs from {n_baskets:,} baskets to {args.out_dir}.")


if __name__ == "__main__":
    main()
