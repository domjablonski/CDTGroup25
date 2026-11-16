"""RQ2 – recipe similarity with MinHash + LSH, "recipes like this one" (owner: R2, docs/CONTRACT.md 3.4.2).

Our own implementation (NumPy, Numba if needed). datasketch is only used in comparison tests.
Outputs: config.NEIGHBOURS (R2 -> R5) and config.DUPLICATES (R2 -> R3, R5).
"""
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from foodcom import config


def jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity, used as ground truth on samples."""
    return len(a & b) / len(a | b) if a or b else 0.0


def stable_hash(token: str) -> int:
    """32-bit hash that is identical on every machine and run (CRC32)."""
    return zlib.crc32(token.encode("utf-8"))


def s_curve(s: float | np.ndarray, bands: int, rows: int) -> float | np.ndarray:
    """Probability that two sets with Jaccard s become a candidate pair: 1 - (1 - s^r)^b."""
    return 1 - (1 - np.asarray(s) ** rows) ** bands


@dataclass
class LSHIndex:
    """Everything a query needs. Built once by build_index(), reused by every search."""
    recipe_ids: np.ndarray                      # position -> recipe_id
    sets: list                                  # position -> set of hashed shingles
    signatures: np.ndarray                      # (n_hashes, n_recipes), uint32
    bands: int
    rows: int
    seed: int
    buckets: list = field(default_factory=list)  # one dict per band: band key -> list of positions


def shingles(recipe: pd.Series, mode: str = "ingredients", k: int = 5) -> set[int]:
    """Hashed shingles of one recipe.

    mode="ingredients": the set of normalised ingredients (recipe["ingredients_norm"]).
    mode="steps": k-shingles (k words) of " ".join(recipe["steps"]).lower().
    Returns a set of 32-bit integers from stable_hash() (never Python's hash(): it changes
    between processes).
    """
    raise NotImplementedError("R2")


def minhash_signatures(sets: list[set[int]], n_hashes: int = 128, seed: int = 42) -> np.ndarray:
    """MinHash signature matrix of shape (n_hashes, len(sets)), dtype uint32.

    h_i(x) = (a_i * x + b_i) mod (2^31 - 1), a_i in [1, p), b_i in [0, p) from
    np.random.default_rng(seed); empty set -> p. Computed in int64/uint64 (no overflow).
    """
    raise NotImplementedError("R2")


def lsh_candidate_pairs(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    """Banding technique: position pairs (i < j) that share at least one band bucket.

    bands * rows must equal signatures.shape[0].
    """
    raise NotImplementedError("R2")


def build_index(recipes: pd.DataFrame, n_hashes: int = 128, bands: int = 16, rows: int = 8,
                seed: int = config.SEED) -> LSHIndex:
    """Shingles (ingredients mode) + signatures + band buckets for every recipe."""
    raise NotImplementedError("R2")


def similar_recipes(index: LSHIndex, ingredients: list[str], k: int = 10) -> pd.DataFrame:
    """"Recipes like this one": the k recipes most similar to a list of (normalised) ingredients.

    Candidates come from the band buckets only; they are ranked by exact Jaccard.
    Returns columns [recipe_id, similarity], best first (fewer than k rows if few candidates).
    """
    raise NotImplementedError("R2")


def neighbours_table(index: LSHIndex, k: int = 20) -> pd.DataFrame:
    """For every recipe, its k most similar other recipes among the LSH candidates.

    schemas.NEIGHBOURS = [recipe_id, neighbour_id, similarity, rank] (rank 1 = most similar),
    similarity = exact Jaccard. Written to config.NEIGHBOURS for R5.
    """
    raise NotImplementedError("R2")


def find_duplicates(index: LSHIndex, threshold: float = config.DUPLICATE_JACCARD) -> pd.DataFrame:
    """Near-duplicate recipes: candidate pairs with exact Jaccard >= threshold.

    schemas.DUPLICATES = [recipe_id, canonical_id, similarity]: one row per non-canonical recipe.
    Groups = connected components of the accepted pairs (union-find); canonical_id = the oldest
    recipe of the group (smallest "submitted", ties -> smallest recipe_id). Written to
    config.DUPLICATES for R3 (baskets) and R5 (train/test split).
    """
    raise NotImplementedError("R2")
