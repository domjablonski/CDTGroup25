import pandas as pd
import numpy as np
import math
from itertools import combinations
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

#Read the data
df = pd.read_parquet(PROCESSED_DIR / "recipes.parquet")

#Collect the unique tags
#If one want to use ingredients instead of tags
#one can simply replace the column name in the following line
all_tags = set()

for tags in df["tags"]:
    all_tags.update(tags)

#Sort alphabetically for consistent indexing
sorted_tags = sorted(all_tags)

#Map tags to indices and vice versa
ingredient_to_id = {
    ingredient: idx
    for idx, ingredient in enumerate(sorted_tags)
}

id_to_ingredient = {
    idx: ingredient
    for ingredient, idx in ingredient_to_id.items()
}

baskets = []

for tags in df["tags"]:
    item_ids = {
        ingredient_to_id[item]
        for item in tags
    }
    baskets.append(sorted(item_ids))

#Starting parameters for PCY algorithm
#Adjusted for tags instead of ingredients
min_support = 10000  # Minimum number of occurrences for an ingredient to be considered frequent
num_buckets = 100000  # Number of buckets for hashing pairs of tags (can be increased)
#Additionally we need a hash-function to map pairs of tags to buckets.
def hash_pair(i, j):
    return hash((min(i, j), max(i, j))) % num_buckets

#Here the actual PCY algo starts
num_items = len(ingredient_to_id)
singleton_counts = np.zeros(num_items, dtype=np.int64)
# Count occurrences of hashed ingredient pairs
bucket_counts = np.zeros(num_buckets, dtype=np.int64)

for basket in baskets:
    # Count individual tags
    for item in basket:
        singleton_counts[item] += 1
    # Hash every pair into a bucket
    for item1, item2 in combinations(basket, 2):
        bucket = hash_pair(item1, item2)
        bucket_counts[bucket] += 1

# Create the PCY bitmap
frequent_buckets = bucket_counts >= min_support
# Identify frequent individual tags
frequent_items = singleton_counts >= min_support

#PCY Pass 2
pair_counts = defaultdict(int)

for basket in baskets:
    # Keep only frequent individual tags
    frequent_basket = [
        item for item in basket
        if frequent_items[item]
    ]
    # Generate pairs from the remaining tags
    for i, j in combinations(frequent_basket, 2):
        bucket = hash_pair(i, j)
        # Only count pairs hashing to frequent buckets
        if frequent_buckets[bucket]:
            pair_counts[(i, j)] += 1

frequent_pairs = {
    pair: count
    for pair, count in pair_counts.items()
    if count >= min_support
}

#Frequent triples
# Set of all frequent pairs
frequent_pair_set = set(frequent_pairs.keys())
# Group pairs by their first ingredient
pairs_by_first = defaultdict(list)
for i, j in frequent_pair_set:
    pairs_by_first[i].append(j)

# Generate candidate triples
candidate_triples = set()
for i, items in pairs_by_first.items():
    items = sorted(items)

    for j, k in combinations(items, 2):
        # Check whether the third pair is also frequent
        if (j, k) in frequent_pair_set:
            candidate_triples.add((i, j, k))

triple_counts = defaultdict(int)
for basket in baskets:

    # Keep only frequent individual tags
    frequent_basket = [
        item for item in basket
        if frequent_items[item]
    ]

    # Generate all triples in this recipe
    for triple in combinations(frequent_basket, 3):

        # Count only candidate triples
        if triple in candidate_triples:
            triple_counts[triple] += 1


frequent_triples = {
    triple: count
    for triple, count in triple_counts.items()
    if count >= min_support
}


#Map frequent singles back to ingredient names
frequent_singles_named = {
    id_to_ingredient[i]: int(singleton_counts[i])
    for i in np.where(frequent_items)[0]
}

# Map frequent pairs back to ingredient names
frequent_pairs_named = {
    (id_to_ingredient[i], id_to_ingredient[j]): count
    for (i, j), count in frequent_pairs.items()
}

# Map frequent triples back to ingredient names
frequent_triples_named = {
    (id_to_ingredient[i], id_to_ingredient[j], id_to_ingredient[k]): count
    for (i, j, k), count in frequent_triples.items()
}

#Save frequent singles to a file
pd.DataFrame([
    {"ingredient": i, "support": count}
    for i, count in frequent_singles_named.items()
]).to_parquet(PROCESSED_DIR / "frequent_tags_singles.parquet", index=False)

#Save frequent pairs to a file
pd.DataFrame([
    {"ingredient1": i, "ingredient2": j, "support": count}
    for (i, j), count in frequent_pairs_named.items()
]).to_parquet(PROCESSED_DIR / "frequent_tags_pairs.parquet", index=False)

#Save frequent triples to a file
pd.DataFrame([
    {"ingredient1": i, "ingredient2": j, "ingredient3": k, "support": count}
    for (i, j, k), count in frequent_triples_named.items()
]).to_parquet(PROCESSED_DIR / "frequent_tags_triples.parquet", index=False)
