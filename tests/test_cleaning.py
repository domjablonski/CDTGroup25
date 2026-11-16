import importlib

import pandas as pd
import pytest

from foodcom import cleaning, config, lsh

RAW_RECIPES = pd.DataFrame({
    "name": ["arriba   baked squash", "garlic bread", "garlic bread"],
    "id": [1, 2, 3],
    "minutes": [55, config.INT32_MAX, 0],
    "contributor_id": [10, 11, 11],
    "submitted": ["2005-09-16", "2008-01-02", "2009-03-04"],
    "tags": ["['mexican', 'easy']", "['italian', 'asian']", "['easy']"],
    "nutrition": ["[51.5, 0.0, 13.0, 0.0, 2.0, 0.0, 4.0]",
                  "[6000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
                  "[100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"],
    "n_steps": [2, 1, 1],
    "steps": ["['cut', 'bake']", "['toast']", "['toast']"],
    "description": ["autumn", None, "copy"],
    "ingredients": ["['Butter', 'garlic cloves']", "['garlic', 'all-purpose flour']", "['garlic', 'flour']"],
    "n_ingredients": [2, 2, 2],
})
RAW_INTERACTIONS = pd.DataFrame({
    "user_id": [100, 100, 101, 102],
    "recipe_id": [1, 1, 2, 999],
    "date": ["2010-01-01", "2010-01-02", "2011-05-05", "2012-07-07"],
    "rating": [5, 4, 0, 3],
    "review": ["yum", "again", None, "ok"],
})


@pytest.fixture
def recipes():
    return cleaning.clean_recipes(RAW_RECIPES.copy())


def test_lists_and_nutrition_are_parsed(recipes):
    assert recipes.loc[0, "tags"] == ["mexican", "easy"]
    assert set(config.NUTRITION_COLS) <= set(recipes.columns) and "nutrition" not in recipes.columns
    assert recipes.loc[0, "calories"] == 51.5
    assert recipes.loc[0, "name"] == "arriba baked squash"


def test_suspicious_values_are_flagged(recipes):
    assert pd.isna(recipes.loc[1, "minutes"])                      # overflow value removed
    assert recipes["flag_minutes"].tolist() == [False, True, True]  # overflow and 0 minutes
    assert recipes["flag_calories"].tolist() == [False, True, False]
    assert recipes["flag_dup_name"].tolist() == [False, True, True]


def test_ingredients_are_normalised(recipes):
    assert recipes.loc[0, "ingredients_norm"] == ["butter", "garlic"]
    assert recipes.loc[1, "ingredients_norm"] == recipes.loc[2, "ingredients_norm"] == ["flour", "garlic"]


def test_cuisine_label_prefers_specific_tag(recipes):
    assert recipes["cuisine"].tolist()[:2] == ["mexican", "italian"]
    assert pd.isna(recipes.loc[2, "cuisine"])


def test_calorie_level_tertiles_skip_flagged_recipes(mini_recipes):
    levels = mini_recipes.set_index("recipe_id")["calorie_level"]
    assert pd.isna(levels[107])                                    # 6,200 kcal: flagged, no level
    kept = levels.drop(107)
    assert set(kept) == {0, 1, 2}
    assert levels[101] == 0 and levels[106] == 2                   # 51.5 kcal low, 420 kcal high
    assert kept.value_counts().max() - kept.value_counts().min() <= 1   # 7 recipes -> 3/2/2 or 2/3/2


def test_interactions(recipes):
    inter = cleaning.clean_interactions(RAW_INTERACTIONS.copy(), recipes["recipe_id"])
    assert inter["rated"].tolist() == [True, True, False, True]
    assert inter["flag_unknown_recipe"].tolist() == [False, False, False, True]
    assert inter["flag_duplicate"].tolist() == [False, True, False, False]
    assert inter.loc[2, "review"] == ""


def test_sample_keeps_interactions_consistent(recipes):
    inter = cleaning.clean_interactions(RAW_INTERACTIONS.copy(), recipes["recipe_id"])
    r_s, i_s = cleaning.make_sample(recipes, inter, frac=0.5, seed=0)
    assert set(i_s["recipe_id"]) <= set(r_s["recipe_id"])


def test_lsh_helpers():
    assert lsh.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert lsh.s_curve(1.0, bands=20, rows=5) == 1.0
    assert lsh.s_curve(0.0, bands=20, rows=5) == 0.0


@pytest.mark.parametrize("module", ["download", "lsh", "itemsets", "graphs", "classify"])
def test_modules_import(module):
    importlib.import_module(f"foodcom.{module}")
