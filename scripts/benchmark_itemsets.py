"""Benchmark exact A-Priori and PCY mining on identical ingredient baskets."""

import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
from itertools import combinations
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .recipe_data import DEFAULT_SEED, read_recipes, staged_outputs
else:
    from recipe_data import DEFAULT_SEED, read_recipes, staged_outputs

BENCH_COLUMNS = [
    "method", "support", "support_count", "buckets", "elapsed_sec", "peak_memory_mb",
    "candidate_pairs_counted", "frequent_singles", "frequent_pairs", "exact_match_with_apriori",
]
RULE_COLUMNS = [
    "group", "min_support", "antecedent", "consequent", "pair_count", "support", "confidence", "lift",
]


@dataclass(frozen=True)
class Baskets:
    vocabulary: tuple[str, ...]
    transactions: list[tuple[int, ...]]


@dataclass(frozen=True)
class Itemsets:
    singles: dict[int, int]
    pairs: dict[tuple[int, int], int]
    triples: dict[tuple[int, int, int], int]
    candidate_pairs_counted: int


def encode_baskets(ingredients: Iterable[Iterable[str]]) -> Baskets:
    """Give ingredients alphabetical IDs and count each one at most once per recipe."""
    baskets = [tuple(basket) for basket in ingredients]
    vocabulary = tuple(sorted({item for basket in baskets for item in basket}))
    item_ids = {item: position for position, item in enumerate(vocabulary)}
    transactions = [tuple(sorted({item_ids[item] for item in basket})) for basket in baskets]
    return Baskets(vocabulary, transactions)


def parse_support(value: str) -> Decimal:
    try:
        support = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("Support must be a fraction in (0, 1].") from exc
    if not support.is_finite() or not 0 < support <= 1:
        raise argparse.ArgumentTypeError("Support must be a fraction in (0, 1].")
    return support


def support_count(support: Decimal | float, n_baskets: int) -> int:
    """Round the decimal support fraction up to a whole basket count."""
    if n_baskets < 1:
        raise ValueError("At least one basket is required.")
    return ceil(parse_support(str(support)) * n_baskets)


def _count_triples(
    baskets: Baskets,
    singles: dict[int, int],
    pairs: dict[tuple[int, int], int],
    minimum: int,
) -> dict[tuple[int, int, int], int]:
    next_items = defaultdict(list)
    for a, b in pairs:
        next_items[a].append(b)
    candidates = {
        (a, b, c)
        for a, partners in next_items.items()
        for b, c in combinations(sorted(partners), 2)
        if (b, c) in pairs
    }
    if not candidates:
        return {}
    counts = Counter(
        triple
        for basket in baskets.transactions
        for triple in combinations((item for item in basket if item in singles), 3)
        if triple in candidates
    )
    return {triple: count for triple, count in counts.items() if count >= minimum}


def apriori(baskets: Baskets, minimum: int, *, include_triples: bool = False) -> Itemsets:
    """Level-wise baseline with anti-monotonic support pruning."""
    if minimum < 1:
        raise ValueError("Minimum support count must be positive.")
    singleton_counts = Counter(item for basket in baskets.transactions for item in basket)
    singles = {item: count for item, count in singleton_counts.items() if count >= minimum}
    pair_counts = Counter(
        pair
        for basket in baskets.transactions
        for pair in combinations((item for item in basket if item in singles), 2)
    )
    pairs = {pair: count for pair, count in pair_counts.items() if count >= minimum}
    triples = _count_triples(baskets, singles, pairs, minimum) if include_triples else {}
    return Itemsets(singles, pairs, triples, len(pair_counts))


def _pair_bucket(a: int, b: int, buckets: int) -> int:
    # Encoded baskets are sorted, so a < b. Same triangular hash as Andreas's PCY.
    return (b * (b - 1) // 2 + a) % buckets


def run_pcy(
    baskets: Baskets, minimum: int, buckets: int, *, include_triples: bool = False
) -> Itemsets:
    """Standalone equivalent of Andreas's structured PCY at d1eedf4."""
    if minimum < 1 or buckets < 1:
        raise ValueError("Minimum support count and bucket count must be positive.")
    singleton_counts = np.zeros(len(baskets.vocabulary), dtype=np.int64)
    bucket_counts = np.zeros(buckets, dtype=np.int64)
    for basket in baskets.transactions:
        for item in basket:
            singleton_counts[item] += 1
        for a, b in combinations(basket, 2):
            bucket_counts[_pair_bucket(a, b, buckets)] += 1

    singles = {int(item): int(singleton_counts[item]) for item in np.flatnonzero(singleton_counts >= minimum)}
    frequent_buckets = bucket_counts >= minimum
    pair_counts = Counter()
    for basket in baskets.transactions:
        frequent_in_basket = (item for item in basket if item in singles)
        for a, b in combinations(frequent_in_basket, 2):
            if frequent_buckets[_pair_bucket(a, b, buckets)]:
                pair_counts[a, b] += 1
    pairs = {pair: count for pair, count in pair_counts.items() if count >= minimum}
    triples = _count_triples(baskets, singles, pairs, minimum) if include_triples else {}
    return Itemsets(singles, pairs, triples, len(pair_counts))


def association_rules(
    baskets: Baskets, itemsets: Itemsets, support: float, min_confidence: float = 0.3, limit: int = 20
) -> list[dict]:
    """Rank both directions of frequent pairs by lift, confidence, and support."""
    n_baskets = len(baskets.transactions)
    rules = []
    for (a, b), count in itemsets.pairs.items():
        lift = count * n_baskets / (itemsets.singles[a] * itemsets.singles[b])
        for antecedent, consequent in ((a, b), (b, a)):
            confidence = count / itemsets.singles[antecedent]
            if confidence >= min_confidence:
                rules.append({
                    "group": "all",
                    "min_support": support,
                    "antecedent": baskets.vocabulary[antecedent],
                    "consequent": baskets.vocabulary[consequent],
                    "pair_count": count,
                    "support": count / n_baskets,
                    "confidence": confidence,
                    "lift": lift,
                })
    rules.sort(key=lambda row: (
        -row["lift"], -row["confidence"], -row["support"], row["antecedent"], row["consequent"],
    ))
    return rules[:limit]


def _measure(mine: Callable[[], Itemsets]) -> tuple[Itemsets, float, float]:
    tracemalloc.start()
    try:
        start = time.perf_counter()
        result = mine()
        elapsed = time.perf_counter() - start
        peak_memory_mb = tracemalloc.get_traced_memory()[1] / 1_000_000
    finally:
        tracemalloc.stop()
    return result, elapsed, peak_memory_mb


def benchmark(
    baskets: Baskets,
    supports: list[Decimal],
    bucket_counts: list[int],
    *,
    include_triples: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time mining with tracemalloc on identical prepared inputs.

    Candidate counts measure distinct pair counters kept before support filtering.
    """
    runs = []
    rules = []
    for support in supports:
        minimum = support_count(support, len(baskets.transactions))
        baseline = None
        for buckets in (None, *bucket_counts):
            if buckets is None:
                method = "apriori"
                mine = partial(apriori, baskets, minimum, include_triples=include_triples)
            else:
                method = "pcy"
                mine = partial(run_pcy, baskets, minimum, buckets, include_triples=include_triples)
            counts, elapsed, peak_memory_mb = _measure(mine)
            frequent = (counts.singles, counts.pairs, counts.triples)
            if buckets is None:
                baseline = frequent
                rules.extend(association_rules(baskets, counts, float(support)))
            runs.append({
                "method": method,
                "support": float(support),
                "support_count": minimum,
                "buckets": buckets,
                "elapsed_sec": elapsed,
                "peak_memory_mb": peak_memory_mb,
                "candidate_pairs_counted": counts.candidate_pairs_counted,
                "frequent_singles": len(counts.singles),
                "frequent_pairs": len(counts.pairs),
                "exact_match_with_apriori": frequent == baseline,
            })
    timings = pd.DataFrame(runs, columns=BENCH_COLUMNS).astype({"buckets": "Int64"})
    return timings, pd.DataFrame(rules, columns=RULE_COLUMNS)


def _parse_buckets(value: str) -> list[int]:
    try:
        buckets = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Buckets must be comma-separated positive integers.") from exc
    if any(count < 1 for count in buckets):
        raise argparse.ArgumentTypeError("Bucket counts must be positive.")
    return buckets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True, help="Recipe CSV or Parquet")
    parser.add_argument("--ingredient-col", default="ingredients_v2")
    parser.add_argument("--supports", type=lambda value: [parse_support(s) for s in value.split(",")],
                        default="0.001,0.005,0.01")
    parser.add_argument("--buckets", type=_parse_buckets, default="50000,100000,250000")
    parser.add_argument("--sample-size", type=int, help="Use at most this many recipe rows")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--triples", action="store_true", help="Mine triples in both algorithms")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    recipes = read_recipes(args.recipes, args.ingredient_col, args.sample_size, args.seed)
    baskets = encode_baskets(recipes[args.ingredient_col])
    timings, rules = benchmark(baskets, args.supports, args.buckets, include_triples=args.triples)
    with staged_outputs(args.out_dir) as stage:
        timings.to_csv(stage / "itemsets_bench.csv", index=False)
        rules.to_csv(stage / "rules_top.csv", index=False)
    print(f"Benchmarked {len(baskets.transactions):,} baskets across {len(timings)} runs in {args.out_dir}.")
    if not timings["exact_match_with_apriori"].all():
        raise SystemExit("PCY disagreed with A-Priori. See itemsets_bench.csv.")


if __name__ == "__main__":
    main()
