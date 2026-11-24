"""Shared train / validation / test split for cuisine prediction (R4 graph communities, R5 XGBoost).

Both models must be trained and evaluated on exactly the same recipes, so the split is written once to
splits/cuisine_split.csv (id, cuisine, split) and everybody reads that file.

Rules
  * label: cuisine = first of the 16 cuisine tags found in the recipe tags (most specific first)
  * classes: cuisines with at least 1,500 labelled recipes (11 classes); other recipes are left out
  * recipes without any ingredient are left out
  * test = 20 %, then validation = 20 % of the rest (16 %), train = the remaining 64 %
  * stratified by cuisine, and recipes with exactly the same ingredient set stay in the same part
    (otherwise a copy in train would give the answer for its twin in test)
  * seed 42; use val to choose hyperparameters, then refit on train + val and evaluate once on test

Usage (repo root, after preprocessing): python splits/make_cuisine_split.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
MIN_RECIPES = 1500
CUISINE_TAGS = ["chinese", "thai", "japanese", "korean", "indian", "italian", "mexican", "greek",
                "french", "spanish", "german", "middle-eastern", "caribbean", "southern-united-states",
                "african", "asian"]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "cuisine_split.csv"


def cuisine_label(tags) -> str | None:
    """First tag of CUISINE_TAGS present in the recipe tags, or None."""
    tag_set = set(tags)
    return next((c for c in CUISINE_TAGS if c in tag_set), None)


def _one_fold(df: pd.DataFrame, frac: float, seed: int) -> pd.Series:
    """Boolean mask of one stratified, grouped fold holding about `frac` of the rows."""
    folds = StratifiedGroupKFold(n_splits=round(1 / frac), shuffle=True, random_state=seed)
    _, held_out = next(folds.split(df, df["cuisine"], groups=df["group"]))
    mask = pd.Series(False, index=df.index)
    mask.iloc[held_out] = True
    return mask


def make_split(recipes: pd.DataFrame, min_recipes: int = MIN_RECIPES, test_frac: float = 0.2,
               val_frac: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    """recipes (id, tags, ingredients_v2) -> DataFrame id, cuisine, split in {train, val, test}."""
    df = recipes[["id"]].copy()
    df["cuisine"] = recipes["tags"].map(lambda t: cuisine_label(list(t)))
    ingredients = recipes["ingredients_v2"].map(lambda x: sorted(set(x)))
    df["group"] = ingredients.map("|".join)
    df = df[df["cuisine"].notna() & (ingredients.map(len) > 0)]
    counts = df["cuisine"].value_counts()
    df = df[df["cuisine"].isin(counts[counts >= min_recipes].index)].sort_values("id", ignore_index=True)

    df["split"] = "train"
    is_test = _one_fold(df, test_frac, seed)
    df.loc[is_test, "split"] = "test"
    rest = df[~is_test]
    df.loc[rest.index[_one_fold(rest, val_frac, seed).to_numpy()], "split"] = "val"
    return df[["id", "cuisine", "split"]]


def main() -> None:
    root = HERE.parent
    matches = sorted(root.glob("data*/**/processed/recipes.parquet"))
    if not matches:
        raise FileNotFoundError(f"no data*/**/processed/recipes.parquet under {root}: run preprocessing first")
    recipes = pd.read_parquet(matches[0], columns=["id", "tags", "ingredients_v2"])
    split = make_split(recipes)
    split.to_csv(OUTPUT, index=False)
    table = pd.crosstab(split["cuisine"], split["split"], margins=True, margins_name="total")
    print(f"{len(split):,} labelled recipes in {split['cuisine'].nunique()} cuisines -> {OUTPUT}")
    print(table[["train", "val", "test", "total"]].sort_values("total", ascending=False).to_string())


if __name__ == "__main__":
    main()
