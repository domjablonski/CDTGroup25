"""RQ1 – ingredient pairing with A-Priori and PCY (owner: R3, docs/CONTRACT.md 3.4.3).

Our own implementation; mlxtend is only used in comparison tests on the sample.
Baskets = recipe["ingredients_norm"] of recipes that are NOT near-duplicates (config.DUPLICATES, R2).
Outputs: config.PAIRS (R3 -> R4, schemas.PAIRS) and config.RULES (schemas.RULES).
"""
import pandas as pd

Itemset = frozenset[str]


def baskets(recipes: pd.DataFrame, duplicates: pd.DataFrame | None = None) -> list[list[str]]:
    """ingredients_norm of every recipe whose recipe_id is not in duplicates.recipe_id."""
    raise NotImplementedError("R3")


def apriori(baskets: list[list[str]], min_support: float, max_size: int = 3) -> dict[Itemset, int]:
    """Frequent itemsets (size 1..max_size) with their counts.

    min_support is a fraction: an itemset is frequent if count >= ceil(min_support * len(baskets)).
    """
    raise NotImplementedError("R3")


def pcy(baskets: list[list[str]], min_support: float, n_buckets: int = 1_000_003) -> dict[Itemset, int]:
    """Frequent items and pairs with PCY (pass 1: item counts + pair hash buckets in a NumPy
    int32 array, then a bitmap of frequent buckets; pass 2: count only pairs of frequent items
    whose bucket is frequent). Bucket hash from lsh.stable_hash of the two sorted items.

    Must return exactly the itemsets of size <= 2 that apriori(max_size=2) returns.
    """
    raise NotImplementedError("R3")


def association_rules(itemsets: dict[Itemset, int], n_baskets: int,
                      min_confidence: float = 0.3, group: str = "all") -> pd.DataFrame:
    """Rules X -> Y from itemsets of size >= 2, schemas.RULES columns:
    antecedent / consequent (sorted lists of str), support = c(X∪Y)/N, confidence = c(X∪Y)/c(X),
    lift = confidence / (c(Y)/N), group."""
    raise NotImplementedError("R3")


def pair_table(itemsets: dict[Itemset, int], n_baskets: int) -> pd.DataFrame:
    """Every frequent pair with the counts R4 needs for PMI: schemas.PAIRS =
    [item_a, item_b, pair_count, count_a, count_b, n_baskets], item_a < item_b."""
    raise NotImplementedError("R3")


def rules_by_group(recipes: pd.DataFrame, group_col: str, min_support: float) -> pd.DataFrame:
    """association_rules() computed separately for each value of group_col (e.g. "cuisine"),
    with group = that value."""
    raise NotImplementedError("R3")
