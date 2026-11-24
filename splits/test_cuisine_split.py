"""Tests for the shared cuisine split (R4 + R5). Run: pytest splits -q"""
import numpy as np
import pandas as pd
import pytest

from make_cuisine_split import CUISINE_TAGS, cuisine_label, make_split


def toy_recipes(seed=0):
    """3 cuisines with 400 / 300 / 50 recipes, 30 recipes without cuisine, 20 without ingredients,
    and groups of exact copies (same ingredient set) inside every cuisine."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 0
    for cuisine, n in (("italian", 400), ("mexican", 300), ("korean", 50), (None, 30)):
        for i in range(n):
            if i % 10 < 3 and i > 0:          # 30 % of the recipes copy the previous ingredient set
                ingredients = rows[-1]["ingredients_v2"]
            else:
                ingredients = list(rng.choice([f"ing{k}" for k in range(60)], size=6, replace=False))
            tags = ["easy"] + ([cuisine] if cuisine else [])
            rows.append({"id": rid, "tags": tags, "ingredients_v2": ingredients})
            rid += 1
    for _ in range(20):
        rows.append({"id": rid, "tags": ["italian"], "ingredients_v2": []})
        rid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def split():
    return make_split(toy_recipes(), min_recipes=100, test_frac=0.2, val_frac=0.2, seed=42)


def test_cuisine_label_most_specific_tag():
    assert cuisine_label(["asian", "chinese"]) == "chinese"
    assert cuisine_label(["easy"]) is None
    assert len(CUISINE_TAGS) == 16


def test_split_columns_and_values(split):
    assert list(split.columns) == ["id", "cuisine", "split"]
    assert set(split["split"]) == {"train", "val", "test"}


def test_small_cuisines_unlabelled_and_empty_recipes_are_left_out(split):
    assert set(split["cuisine"]) == {"italian", "mexican"}  # korean: 50 < 100 recipes
    recipes = toy_recipes()
    empty_ids = set(recipes.loc[recipes["ingredients_v2"].map(len) == 0, "id"])
    assert not empty_ids & set(split["id"])
    assert len(split) == 700 and split["id"].is_unique


def test_copies_always_on_the_same_side(split):
    recipes = toy_recipes().set_index("id")
    key = split["id"].map(lambda i: "|".join(sorted(recipes.loc[i, "ingredients_v2"])))
    assert split.groupby(key)["split"].nunique().max() == 1


def test_proportions_and_stratification(split):
    shares = split["split"].value_counts(normalize=True)
    assert shares["test"] == pytest.approx(0.20, abs=0.04)
    assert shares["val"] == pytest.approx(0.16, abs=0.04)  # 20 % of the remaining 80 %
    by_class = pd.crosstab(split["cuisine"], split["split"], normalize="index")
    assert by_class["test"].between(0.14, 0.26).all()


def test_split_is_reproducible_and_depends_on_the_seed():
    a = make_split(toy_recipes(), min_recipes=100, seed=42)
    b = make_split(toy_recipes(), min_recipes=100, seed=42)
    c = make_split(toy_recipes(), min_recipes=100, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])
