"""Parse, clean and save the raw data (owner: R1).

Run once after downloading:  python -m foodcom.cleaning
Then everyone loads the data with load_recipes() / load_interactions().

Suspicious values are flagged (flag_* columns), not dropped: each role decides what to exclude
and says so in the report.
"""
import ast
import re

import pandas as pd

from foodcom import config

# Obvious synonyms only. R1 extends this after looking at the most frequent ingredients;
# ambiguous names ("pepper", "unsalted butter") are left as they are on purpose.
INGREDIENT_SYNONYMS = {
    "garlic cloves": "garlic",
    "garlic clove": "garlic",
    "all-purpose flour": "flour",
    "all purpose flour": "flour",
    "granulated sugar": "sugar",
    "white sugar": "sugar",
    "vanilla extract": "vanilla",
    "pure vanilla extract": "vanilla",
    "egg": "eggs",
    "large eggs": "eggs",
}
_SPACES = re.compile(r"\s+")


def normalize_ingredient(name: str) -> str:
    s = _SPACES.sub(" ", str(name).lower()).strip()
    return INGREDIENT_SYNONYMS.get(s, s)


def cuisine_label(tags: list[str]) -> str | None:
    """First cuisine tag of config.CUISINE_TAGS found in the recipe tags, or None."""
    tag_set = set(tags)
    return next((c for c in config.CUISINE_TAGS if c in tag_set), None)


def calorie_level(recipes: pd.DataFrame) -> pd.Series:
    """Label of RQ4: calorie tertiles (0 low, 1 medium, 2 high) of the recipes without flag_calories.

    Flagged recipes (> config.CALORIES_MAX kcal) get <NA>. Computed on the full data before the sample
    is drawn, so the thresholds are the same everywhere. dtype Int8.
    """
    ok = ~recipes["flag_calories"]
    levels = pd.qcut(recipes.loc[ok, "calories"], q=config.CALORIE_LEVELS, labels=False, duplicates="drop")
    return levels.reindex(recipes.index).astype("Int8")


def clean_recipes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"id": "recipe_id"})
    df["name"] = df["name"].fillna("").map(lambda s: _SPACES.sub(" ", s).strip())
    for col in ("tags", "steps", "ingredients"):
        df[col] = df[col].map(ast.literal_eval)
    nutrition = pd.DataFrame(df.pop("nutrition").map(ast.literal_eval).tolist(),
                             columns=config.NUTRITION_COLS, index=df.index)
    df = pd.concat([df, nutrition], axis=1)
    df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")

    df["minutes"] = df["minutes"].astype("float64").where(df["minutes"] != config.INT32_MAX)
    df["flag_minutes"] = df["minutes"].isna() | (df["minutes"] == 0) | (df["minutes"] > config.MINUTES_MAX)
    df["flag_calories"] = df["calories"] > config.CALORIES_MAX
    df["flag_dup_name"] = df["name"].ne("") & df["name"].duplicated(keep=False)
    df["calorie_level"] = calorie_level(df)

    df["ingredients_norm"] = df["ingredients"].map(lambda xs: sorted({normalize_ingredient(x) for x in xs}))
    df["cuisine"] = df["tags"].map(cuisine_label)
    return df


def clean_interactions(raw: pd.DataFrame, recipe_ids: pd.Series) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["review"] = df["review"].fillna("")
    df["rated"] = df["rating"] > 0                       # rating 0 = review without a rating
    df["flag_unknown_recipe"] = ~df["recipe_id"].isin(recipe_ids)
    df["flag_duplicate"] = df.duplicated(["user_id", "recipe_id"], keep="first")
    return df


def make_sample(recipes: pd.DataFrame, interactions: pd.DataFrame,
                frac: float = config.SAMPLE_FRAC, seed: int = config.SEED):
    """Random recipes and all their interactions, so both samples stay consistent."""
    recipes_s = recipes.sample(frac=frac, random_state=seed)
    return recipes_s, interactions[interactions["recipe_id"].isin(recipes_s["recipe_id"])]


LIST_COLS = ("tags", "steps", "ingredients", "ingredients_norm")


def load_recipes(sample: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(config.RECIPES_SAMPLE if sample else config.RECIPES)
    for col in LIST_COLS:                 # Parquet gives NumPy arrays back; we want plain lists
        df[col] = df[col].map(list)
    return df


def load_interactions(sample: bool = False) -> pd.DataFrame:
    return pd.read_parquet(config.INTERACTIONS_SAMPLE if sample else config.INTERACTIONS)


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    recipes = clean_recipes(pd.read_csv(config.RAW_RECIPES))
    recipes.to_parquet(config.RECIPES, index=False)
    raw_ingredients = {i for xs in recipes["ingredients"] for i in xs}
    norm_ingredients = {i for xs in recipes["ingredients_norm"] for i in xs}
    print(f"recipes: {len(recipes):,} rows | flag_minutes {recipes.flag_minutes.sum():,} | "
          f"flag_calories {recipes.flag_calories.sum():,} | flag_dup_name {recipes.flag_dup_name.sum():,}")
    print(f"ingredients: {len(raw_ingredients):,} raw labels -> {len(norm_ingredients):,} after normalisation")
    print(f"cuisine label: {recipes.cuisine.notna().sum():,} recipes")
    bounds = recipes.groupby("calorie_level")["calories"].agg(["min", "max", "size"])
    print("calorie levels:", "; ".join(f"{int(k)}: {r['min']:.0f}-{r['max']:.0f} kcal ({r['size']:,})"
                                       for k, r in bounds.iterrows()))

    if not config.RAW_INTERACTIONS.exists():
        print(f"{config.RAW_INTERACTIONS.name} not found: interactions skipped")
        recipes.sample(frac=config.SAMPLE_FRAC, random_state=config.SEED).to_parquet(config.RECIPES_SAMPLE, index=False)
        return
    interactions = clean_interactions(pd.read_csv(config.RAW_INTERACTIONS), recipes["recipe_id"])
    interactions.to_parquet(config.INTERACTIONS, index=False)
    print(f"interactions: {len(interactions):,} rows | rated {interactions.rated.mean():.1%} | "
          f"unknown recipe {interactions.flag_unknown_recipe.sum():,} | duplicates {interactions.flag_duplicate.sum():,}")

    recipes_s, interactions_s = make_sample(recipes, interactions)
    recipes_s.to_parquet(config.RECIPES_SAMPLE, index=False)
    interactions_s.to_parquet(config.INTERACTIONS_SAMPLE, index=False)
    print(f"sample: {len(recipes_s):,} recipes, {len(interactions_s):,} interactions")


if __name__ == "__main__":
    main()
