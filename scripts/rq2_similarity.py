"""Recipe search with CRC32 MinHash, LSH banding, and exact Jaccard reranking."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from heapq import nsmallest
from zlib import crc32

import numpy as np

if __package__:
    from .recipe_data import DEFAULT_SEED, RecipeId
else:
    from recipe_data import DEFAULT_SEED, RecipeId

HASH_PRIME = (1 << 31) - 1


def jaccard(left: Set[str], right: Set[str]) -> float:
    """Exact set Jaccard with similarity zero when both sets are empty."""
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _ingredient_set(ingredients: Iterable[str]) -> frozenset[str]:
    if isinstance(ingredients, str):
        raise TypeError("Pass an iterable of ingredient names.")
    return frozenset(name for name in (item.strip().lower() for item in ingredients) if name)


@dataclass(frozen=True)
class RecipeMatch:
    recipe_id: RecipeId
    similarity: float


def _rank_key(match: RecipeMatch) -> tuple[float, bool, RecipeId]:
    return -match.similarity, isinstance(match.recipe_id, str), match.recipe_id


class RecipeCorpus:
    """Store ingredient sets once for both LSH and exact search."""

    def __init__(self, recipes: Mapping[RecipeId, Iterable[str]]) -> None:
        self.recipe_ids = tuple(recipes)
        self.ingredient_sets = tuple(_ingredient_set(recipes[recipe]) for recipe in self.recipe_ids)
        self.positions = {recipe: position for position, recipe in enumerate(self.recipe_ids)}

    def exact_by_id(self, recipe_id: RecipeId, k: int = 5) -> list[RecipeMatch]:
        """Scan every other recipe in the same candidate population."""
        position = self.positions[recipe_id]
        return self._rank(self.ingredient_sets[position], range(len(self.recipe_ids)), k, exclude=position)

    def _rank(
        self,
        ingredients: frozenset[str],
        candidates: Iterable[int],
        k: int,
        *,
        exclude: int | None = None,
    ) -> list[RecipeMatch]:
        if k < 0:
            raise ValueError("k must be nonnegative.")
        if not k or not ingredients:
            return []
        scored = (
            (position, jaccard(ingredients, self.ingredient_sets[position]))
            for position in candidates if position != exclude
        )
        matches = (RecipeMatch(self.recipe_ids[position], score) for position, score in scored if score > 0)
        return nsmallest(k, matches, key=_rank_key)


class LSHIndex:
    """Seeded MinHash signatures indexed by consecutive bands of equal size."""

    def __init__(
        self, corpus: RecipeCorpus, *, bands: int = 32, rows: int = 4,
        n_hashes: int = 128, seed: int = DEFAULT_SEED,
    ) -> None:
        if bands < 1 or rows < 1 or bands * rows != n_hashes:
            raise ValueError("Positive bands and rows must multiply to n_hashes.")
        self.corpus = corpus
        self.bands = bands
        self.rows = rows

        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, HASH_PRIME, n_hashes, dtype=np.int64)
        self._b = rng.integers(0, HASH_PRIME, n_hashes, dtype=np.int64)
        vocabulary = sorted({item for ingredients in corpus.ingredient_sets for item in ingredients})
        positions = {item: position for position, item in enumerate(vocabulary)}
        hashes = self._hash_values(vocabulary)
        self._signatures = np.full((len(corpus.recipe_ids), n_hashes), HASH_PRIME, dtype=np.uint32)
        self._buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]
        for position, ingredients in enumerate(corpus.ingredient_sets):
            if not ingredients:
                continue
            signature = self._signatures[position]
            signature[:] = hashes[[positions[item] for item in ingredients]].min(axis=0)
            for band, key in enumerate(self._band_keys(signature)):
                self._buckets[band][key].append(position)

    def _hash_values(self, ingredients: Iterable[str]) -> np.ndarray:
        codes = np.fromiter((crc32(item.encode("utf-8")) % HASH_PRIME for item in ingredients), dtype=np.int64)
        # int64 can hold a product of two values below the 31-bit prime.
        return (codes[:, None] * self._a + self._b) % HASH_PRIME

    def _band_keys(self, signature: np.ndarray) -> Iterable[bytes]:
        for start in range(0, len(signature), self.rows):
            yield signature[start:start + self.rows].tobytes()

    def _candidates(self, signature: np.ndarray) -> set[int]:
        candidates = set()
        for bucket, key in zip(self._buckets, self._band_keys(signature)):
            candidates.update(bucket.get(key, ()))
        return candidates

    def candidate_positions(self, position: int) -> set[int]:
        """Return the other corpus positions sharing at least one band."""
        candidates = self._candidates(self._signatures[position])
        candidates.discard(position)
        return candidates

    def query_by_id(self, recipe_id: RecipeId, k: int = 5) -> list[RecipeMatch]:
        """Positive matches by Jaccard and recipe ID, excluding the query recipe."""
        position = self.corpus.positions[recipe_id]
        return self.corpus._rank(self.corpus.ingredient_sets[position], self.candidate_positions(position), k)

    def query_ingredients(self, ingredients: Iterable[str], k: int = 5) -> list[RecipeMatch]:
        """Positive matches for a new ingredient list, including unseen terms in Jaccard."""
        query = _ingredient_set(ingredients)
        candidates: Iterable[int] = ()
        if query:
            signature = self._hash_values(query).min(axis=0).astype(np.uint32)
            candidates = self._candidates(signature)
        return self.corpus._rank(query, candidates, k)
