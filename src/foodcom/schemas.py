"""Columns of every Parquet file exchanged between roles (FROZEN: docs/CONTRACT.md 3.3).

A producer must write at least these columns; a consumer may rely on them and nothing else.
Changing this file needs a `contract` PR approved by all five members.
"""
import pandas as pd

from foodcom import config

RECIPES = ["recipe_id", "name", "minutes", "contributor_id", "submitted", "tags", "n_steps", "steps",
           "description", "ingredients", "n_ingredients", "ingredients_norm", "cuisine", "calorie_level",
           *config.NUTRITION_COLS, "flag_minutes", "flag_calories", "flag_dup_name"]
INTERACTIONS = ["user_id", "recipe_id", "date", "rating", "review", "rated",
                "flag_unknown_recipe", "flag_duplicate"]
DUPLICATES = ["recipe_id", "canonical_id", "similarity"]
NEIGHBOURS = ["recipe_id", "neighbour_id", "similarity", "rank"]
PAIRS = ["item_a", "item_b", "pair_count", "count_a", "count_b", "n_baskets"]
RULES = ["antecedent", "consequent", "support", "confidence", "lift", "group"]

SCHEMAS = {"recipes": RECIPES, "interactions": INTERACTIONS, "duplicates": DUPLICATES,
           "neighbours": NEIGHBOURS, "pairs": PAIRS, "rules": RULES}


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Raise ValueError if df lacks a column of schema `name`; return df unchanged otherwise."""
    missing = [c for c in SCHEMAS[name] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    return df
