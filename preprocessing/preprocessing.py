import ast
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
dataset_matches = [
    recipes_path
    for recipes_path in project_root.glob("data*/**/RAW_recipes.csv")
    if (recipes_path.parent / "RAW_interactions.csv").is_file()
]

if len(dataset_matches) != 1:
    raise FileNotFoundError(
        "Expected exactly one dataset containing RAW_recipes.csv and "
        f"RAW_interactions.csv under {project_root}; found {len(dataset_matches)}."
    )

dataset_dir = dataset_matches[0].parent
out_dir = dataset_dir / "processed"
out_dir.mkdir(exist_ok=True)

recipes = pd.read_csv(dataset_dir / "RAW_recipes.csv")
inter = pd.read_csv(dataset_dir / "RAW_interactions.csv", parse_dates=["date"])

print(recipes.shape, inter.shape)

# 1. Missing values
recipes = recipes.dropna(subset=["name"])
recipes["description"] = recipes["description"].fillna("")
inter["review"] = inter["review"].fillna("")

# 2. Duplicates
recipes = recipes.drop_duplicates(subset="id")
inter = inter.drop_duplicates(subset=["user_id", "recipe_id"])

# 3. Parse string-lists into real lists
for col in ["tags", "ingredients", "steps", "nutrition"]:
    recipes[col] = recipes[col].apply(ast.literal_eval)

# 4. Split nutrition into columns
nutr_cols = ["calories", "total_fat", "sugar", "sodium",
             "protein", "sat_fat", "carbs"]
recipes[nutr_cols] = pd.DataFrame(recipes["nutrition"].tolist(), index=recipes.index)
recipes = recipes.drop(columns="nutrition")

# 5. Inspect before filtering outliers and zero ratings
print(recipes[["minutes", "n_steps", "n_ingredients", "calories"]].describe())
print(inter["rating"].value_counts().sort_index())

recipes = recipes[recipes["minutes"].between(1, 1440)]
inter = inter[inter["rating"] > 0]

# 6. Keep only interactions whose recipe survived filtering
inter = inter[inter["recipe_id"].isin(recipes["id"])]

print(recipes.shape, inter.shape)

# 7. Save
recipes.to_parquet(out_dir / "recipes.parquet")
inter.to_parquet(out_dir / "interactions.parquet")