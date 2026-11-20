
import pandas as pd
import numpy as np

from itertools import combinations
from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. Paths and parameters
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MIN_SUPPORT = 1000
NUM_BUCKETS = 100_000


# ============================================================
# 2. Load data and prepare ingredient baskets
# ============================================================

def load_data():
    """Load the recipe dataset."""
    return pd.read_parquet(PROCESSED_DIR / "recipes.parquet")


def prepare_baskets(df):
    """
    Convert recipes into baskets of integer ingredient IDs.

    Each ingredient appears at most once per recipe.
    IDs are assigned alphabetically for consistent indexing.
    """

    all_ingredients = set()

    for ingredients in df["ingredients"]:
        all_ingredients.update(ingredients)

    sorted_ingredients = sorted(all_ingredients)

    ingredient_to_id = {
        ingredient: idx
        for idx, ingredient in enumerate(sorted_ingredients)
    }

    id_to_ingredient = {
        idx: ingredient
        for ingredient, idx in ingredient_to_id.items()
    }

    baskets = []

    for ingredients in df["ingredients"]:
        item_ids = {
            ingredient_to_id[item]
            for item in ingredients
        }
        baskets.append(sorted(item_ids))

    return baskets, ingredient_to_id, id_to_ingredient


# ============================================================
# 3. Pair hashing
# ============================================================

def hash_pair(i, j, num_buckets):
    """
    Deterministic hash function for unordered ingredient pairs.

    The pair is first converted into a unique integer using
    a pairing function, then mapped to a bucket.
    """

    i, j = sorted((i, j))

    pair_number = j * (j - 1) // 2 + i

    return pair_number % num_buckets


# ============================================================
# 4. PCY algorithm
# ============================================================

def run_pcy(baskets, num_items, min_support=1000, num_buckets=100000):

    if min_support < 1:
        raise ValueError("min_support must be at least 1.")

    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1.")

    # --------------------------------------------------------
    # Pass 1: Count single ingredients and hashed pairs
    # --------------------------------------------------------

    singleton_counts = np.zeros(num_items, dtype=np.int64)
    bucket_counts = np.zeros(num_buckets, dtype=np.int64)

    for basket in baskets:

        # Count individual ingredients
        for item in basket:
            singleton_counts[item] += 1

        # Count hashed pairs
        for i, j in combinations(basket, 2):
            bucket = hash_pair(i, j, num_buckets)
            bucket_counts[bucket] += 1

    # Identify frequent single ingredients
    frequent_items = singleton_counts >= min_support

    # Create PCY bitmap
    frequent_buckets = bucket_counts >= min_support

    print("Pass 1 completed.")
    print("Frequent single ingredients:", frequent_items.sum())
    print("Frequent hash buckets:", frequent_buckets.sum())

    # --------------------------------------------------------
    # Pass 2: Count candidate frequent pairs
    # --------------------------------------------------------

    pair_counts = defaultdict(int)

    for basket in baskets:

        # Keep only frequent single ingredients
        frequent_basket = [
            item for item in basket
            if frequent_items[item]
        ]

        for i, j in combinations(frequent_basket, 2):

            bucket = hash_pair(i, j, num_buckets)

            # Only count pairs hashing to frequent buckets
            if frequent_buckets[bucket]:
                pair_counts[(i, j)] += 1

    frequent_pairs = {
        pair: count
        for pair, count in pair_counts.items()
        if count >= min_support
    }

    print("Pass 2 completed.")
    print("Frequent pairs:", len(frequent_pairs))

    # --------------------------------------------------------
    # Generate candidate triples using frequent pairs
    # --------------------------------------------------------

    frequent_pair_set = set(frequent_pairs.keys())

    pairs_by_first = defaultdict(list)

    for i, j in frequent_pair_set:
        pairs_by_first[i].append(j)

    candidate_triples = set()

    for i, items in pairs_by_first.items():

        items = sorted(items)

        for j, k in combinations(items, 2):

            # All three pairs must be frequent
            if (j, k) in frequent_pair_set:
                candidate_triples.add((i, j, k))

    print("Candidate triples:", len(candidate_triples))

    # --------------------------------------------------------
    # Count candidate triples
    # --------------------------------------------------------

    triple_counts = defaultdict(int)

    for basket in baskets:

        frequent_basket = [
            item for item in basket
            if frequent_items[item]
        ]

        for triple in combinations(frequent_basket, 3):

            if triple in candidate_triples:
                triple_counts[triple] += 1

    frequent_triples = {
        triple: count
        for triple, count in triple_counts.items()
        if count >= min_support
    }

    print("Frequent triples:", len(frequent_triples))

    return {
        "singleton_counts": singleton_counts,
        "frequent_items": frequent_items,
        "frequent_pairs": frequent_pairs,
        "candidate_triples": candidate_triples,
        "frequent_triples": frequent_triples
    }


# ============================================================
# 5. Convert results back to ingredient names
# ============================================================

def name_results(results, id_to_ingredient):

    singleton_counts = results["singleton_counts"]
    frequent_items = results["frequent_items"]
    frequent_pairs = results["frequent_pairs"]
    frequent_triples = results["frequent_triples"]

    frequent_singles_named = {
        id_to_ingredient[i]: int(singleton_counts[i])
        for i in np.where(frequent_items)[0]
    }

    frequent_pairs_named = {
        (id_to_ingredient[i], id_to_ingredient[j]): count
        for (i, j), count in frequent_pairs.items()
    }

    frequent_triples_named = {
        (
            id_to_ingredient[i],
            id_to_ingredient[j],
            id_to_ingredient[k]
        ): count
        for (i, j, k), count in frequent_triples.items()
    }

    return (
        frequent_singles_named,
        frequent_pairs_named,
        frequent_triples_named
    )


# ============================================================
# 6. Save results
# ============================================================

def save_results(singles, pairs, triples):

    singles_df = pd.DataFrame([
        {"ingredient": ingredient, "support": count}
        for ingredient, count in singles.items()
    ])

    pairs_df = pd.DataFrame([
        {
            "ingredient1": i,
            "ingredient2": j,
            "support": count
        }
        for (i, j), count in pairs.items()
    ])

    triples_df = pd.DataFrame([
        {
            "ingredient1": i,
            "ingredient2": j,
            "ingredient3": k,
            "support": count
        }
        for (i, j, k), count in triples.items()
    ])

    singles_df.to_parquet(
        PROCESSED_DIR / "frequent_ingredients_singles.parquet",
        index=False
    )

    pairs_df.to_parquet(
        PROCESSED_DIR / "frequent_ingredients_pairs.parquet",
        index=False
    )

    triples_df.to_parquet(
        PROCESSED_DIR / "frequent_ingredients_triples.parquet",
        index=False
    )

    print("Results saved successfully.")


# ============================================================
# 7. Main execution
# ============================================================

def main():

    df = load_data()

    baskets, ingredient_to_id, id_to_ingredient = prepare_baskets(df)

    print("Number of recipes:", len(baskets))
    print("Number of unique ingredients:", len(ingredient_to_id))

    results = run_pcy(
        baskets=baskets,
        num_items=len(ingredient_to_id),
        min_support=MIN_SUPPORT,
        num_buckets=NUM_BUCKETS
    )

    singles, pairs, triples = name_results(
        results,
        id_to_ingredient
    )

    save_results(singles, pairs, triples)


if __name__ == "__main__":
    main()
