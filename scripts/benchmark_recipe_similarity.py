"""Evaluate ingredient LSH candidates against exact Jaccard on a recipe sample."""

import argparse
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

if __package__:
    from .recipe_data import DEFAULT_SEED, RecipeId, read_recipes, recipe_ids, staged_outputs
    from .rq2_similarity import LSHIndex, RecipeCorpus, RecipeMatch
else:
    from recipe_data import DEFAULT_SEED, RecipeId, read_recipes, recipe_ids, staged_outputs
    from rq2_similarity import LSHIndex, RecipeCorpus, RecipeMatch

CONFIGURATIONS = ((64, 2), (32, 4), (16, 8), (8, 16))
THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
EXAMPLE_CONFIGURATION = (32, 4)
QUERY_COUNT = 100
TOP_K = 5


def _exact_score_rows(corpus: RecipeCorpus, block_size: int = 128) -> Iterator[np.ndarray]:
    """Multiply sparse ingredient incidences in blocks to bound the dense working memory."""
    sets = corpus.ingredient_sets
    vocabulary = sorted({item for ingredients in sets for item in ingredients})
    positions = {item: position for position, item in enumerate(vocabulary)}
    columns = np.fromiter(
        (positions[item] for ingredients in sets for item in sorted(ingredients)), dtype=np.int32,
    )
    sizes = np.fromiter(map(len, sets), dtype=np.int32)
    boundaries = np.concatenate(([0], np.cumsum(sizes)))
    matrix = csr_matrix(
        (np.ones(len(columns), dtype=np.int32), columns, boundaries),
        shape=(len(sets), len(vocabulary)),
    )
    transpose = matrix.T.tocsr()
    for start in range(0, len(sets), block_size):
        stop = start + block_size
        shared = (matrix[start:stop] @ transpose).toarray()
        union = sizes[start:stop, None] + sizes[None, :] - shared
        scores = np.divide(shared, union, out=np.zeros(shared.shape), where=union != 0)
        yield from scores


def evaluate_candidates(
    corpus: RecipeCorpus, indexes: list[LSHIndex], thresholds: tuple[float, ...] = THRESHOLDS
) -> pd.DataFrame:
    """Measure raw candidate precision and recall, counting each unordered pair once.

    Ground truth includes every pair meeting the threshold, whether or not it
    was retrieved by LSH. Precision describes candidates before exact reranking.
    """
    if any(index.corpus is not corpus for index in indexes):
        raise ValueError("Every index must use the benchmark corpus.")
    cutoff = np.array(thresholds)
    relevant_counts = np.zeros(len(cutoff), dtype=np.int64)
    candidate_counts = np.zeros(len(indexes), dtype=np.int64)
    true_positives = np.zeros((len(indexes), len(cutoff)), dtype=np.int64)

    for position, scores in enumerate(_exact_score_rows(corpus)):
        relevant_counts += (scores[position + 1:, None] >= cutoff).sum(axis=0)
        for i, index in enumerate(indexes):
            candidates = np.fromiter(
                (other for other in index.candidate_positions(position) if other > position),
                dtype=np.intp,
            )
            candidate_counts[i] += len(candidates)
            true_positives[i] += (scores[candidates, None] >= cutoff).sum(axis=0)

    rows = []
    for i, index in enumerate(indexes):
        candidates = int(candidate_counts[i])
        for t, threshold in enumerate(thresholds):
            positives = int(relevant_counts[t])
            hits = int(true_positives[i, t])
            rows.append({
                "bands": index.bands,
                "rows": index.rows,
                "n_hashes": index.bands * index.rows,
                "threshold": threshold,
                "sample_size": len(corpus.recipe_ids),
                "candidates": candidates,
                "ground_truth_pairs": positives,
                "true_positives": hits,
                "precision": hits / candidates if candidates else 0.0,
                "recall": hits / positives if positives else 0.0,
                "f1": 2 * hits / (candidates + positives) if candidates + positives else 0.0,
            })
    return pd.DataFrame(rows)


def _timed_queries(
    method: str,
    search: Callable[[RecipeId, int], list[RecipeMatch]],
    queries: list[RecipeId],
    n_recipes: int,
    *,
    index_build_sec: float,
    bands: int | None = None,
    rows: int | None = None,
) -> list[dict]:
    timings = []
    for recipe_id in queries:
        start = time.perf_counter()
        matches = search(recipe_id, TOP_K)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        timings.append({
            "method": method,
            "bands": bands,
            "rows": rows,
            "n_recipes": n_recipes,
            "query_recipe_id": recipe_id,
            "k": TOP_K,
            "index_build_sec": index_build_sec,
            "ms_per_query": elapsed_ms,
            "matches": len(matches),
        })
    return timings


def _example_searches(
    corpus: RecipeCorpus, index: LSHIndex, queries: list[RecipeId], names: dict[RecipeId, str]
) -> pd.DataFrame:
    rows = []
    for recipe_id in queries:
        for method, search in (("lsh", index.query_by_id), ("exact", corpus.exact_by_id)):
            example = {
                "method": method,
                "query_recipe_id": recipe_id,
                "query_name": names.get(recipe_id, ""),
            }
            matches = search(recipe_id, TOP_K)
            if not matches:
                rows.append(example | {
                    "rank": None, "matched_recipe_id": None, "matched_name": "", "jaccard": None,
                })
            for rank, match in enumerate(matches, 1):
                rows.append(example | {
                    "rank": rank,
                    "matched_recipe_id": match.recipe_id,
                    "matched_name": names.get(match.recipe_id, ""),
                    "jaccard": match.similarity,
                })
    result = pd.DataFrame(rows)
    # Keep integer recipe IDs intact when some searches have no match.
    result["matched_recipe_id"] = pd.Series(
        [row["matched_recipe_id"] for row in rows], dtype="object",
    )
    return result.astype({"rank": "Int64"})


def benchmark(
    corpus: RecipeCorpus, names: dict[RecipeId, str], seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare every configuration and time the same seeded queries for each method."""
    n_recipes = len(corpus.recipe_ids)
    if not n_recipes:
        raise ValueError("At least one recipe is required for the benchmark.")
    rng = np.random.default_rng(seed)
    queries = [corpus.recipe_ids[i] for i in rng.choice(n_recipes, min(QUERY_COUNT, n_recipes), replace=False)]

    # Ingredient sets are shared preparation. The brute-force scan builds no index.
    timings = _timed_queries("exact", corpus.exact_by_id, queries, n_recipes, index_build_sec=0.0)
    indexes = {}
    for bands, rows in CONFIGURATIONS:
        start = time.perf_counter()
        index = LSHIndex(corpus, bands=bands, rows=rows, seed=seed)
        build_time = time.perf_counter() - start
        indexes[bands, rows] = index
        timings.extend(_timed_queries(
            "lsh", index.query_by_id, queries, n_recipes, index_build_sec=build_time, bands=bands, rows=rows,
        ))

    metrics = evaluate_candidates(corpus, list(indexes.values()))
    latency = pd.DataFrame(timings).astype({"bands": "Int64", "rows": "Int64"})
    examples = _example_searches(corpus, indexes[EXAMPLE_CONFIGURATION], queries[:3], names)
    return metrics, latency, examples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True, help="Recipe CSV or Parquet")
    parser.add_argument("--ingredient-col", default="ingredients_v2")
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    recipes = read_recipes(args.recipes, args.ingredient_col, args.sample_size, args.seed)
    ids = recipe_ids(recipes)
    corpus = RecipeCorpus(dict(zip(ids, recipes[args.ingredient_col])))
    names = dict(zip(ids, recipes["name"].fillna("").astype(str))) if "name" in recipes else {}
    metrics, latency, examples = benchmark(corpus, names, seed=args.seed)
    with staged_outputs(args.out_dir) as stage:
        metrics.to_csv(stage / "lsh_eval.csv", index=False)
        latency.to_csv(stage / "search_latency.csv", index=False)
        examples.to_csv(stage / "similar_examples.csv", index=False)
    print(f"Evaluated {len(metrics)} configurations and thresholds on {len(ids):,} recipes in {args.out_dir}.")


if __name__ == "__main__":
    main()
